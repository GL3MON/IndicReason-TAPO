"""TAPO token advantages and the TRL policy-gradient implementation."""
from .constants import SUPPORTED_TRL_VERSIONS
from .formatting import translation_token_mask
from .rewards import grouped_standardize, correctness_group_metrics
from .modeling import install_vllm_sleep_policy_sync

def make_trainer_class():
    import torch
    import trl
    from trl import GRPOTrainer

    if trl.__version__ not in SUPPORTED_TRL_VERSIONS:
        raise RuntimeError(f"Unsupported TRL {trl.__version__}; expected one of {sorted(SUPPORTED_TRL_VERSIONS)}")

    class TAPOTrainer(GRPOTrainer):
        """Version-checked private scoring hooks; modern TRL owns the token loss."""

        def __init__(self, *args, tapo_alpha=0.25, tapo_log_entropy=True, **kwargs):
            super().__init__(*args, **kwargs)
            if getattr(self, "use_liger_loss", False) or getattr(self, "use_liger_kernel", False) or self.beta != 0:
                raise ValueError("This implementation requires Liger=False, beta=0")
            if self.use_vllm and trl.__version__ != "1.9.2":
                raise ValueError("TAPO with vLLM requires TRL 1.9.2")
            if self.loss_type != "dapo" or self.importance_sampling_level != "token":
                raise ValueError("This implementation requires loss_type=dapo and token importance sampling")
            if len(self.reward_funcs) != 2:
                raise ValueError("Reward order must be [translation_reward, correctness_reward]")
            self.tapo_alpha = tapo_alpha
            self.tapo_log_entropy = tapo_log_entropy
            if not tapo_log_entropy and trl.__version__ == "1.9.2":
                if (self._entropy_bonus_enabled or self.top_entropy_quantile != 1.0
                        or self.off_policy_mask_threshold is not None or self.aux_loss_enabled
                        or self.args.delta is not None):
                    raise ValueError("Fast TAPO loss requires no entropy bonus/filter, auxiliary loss, or off-policy mask")
            self._tapo_rewards = None
            if self.use_vllm and self.args.vllm_enable_sleep_mode:
                install_vllm_sleep_policy_sync(self.vllm_generation)

        def _calculate_rewards(self, *args, **kwargs):
            rewards = super()._calculate_rewards(*args, **kwargs)
            self._tapo_rewards = rewards.detach().clone()  # already gathered across ranks
            return rewards

        def _generate_and_score_completions(self, inputs):
            self._tapo_rewards = None
            batch = super()._generate_and_score_completions(inputs)
            rewards = self._tapo_rewards
            if rewards is None or not torch.isfinite(rewards).all():
                raise RuntimeError("Missing/non-finite TAPO rewards")
            mode = "train" if self.model.training else "eval"
            group_size = self.num_generations if mode == "train" else getattr(self, "num_generations_eval", self.num_generations)
            for name, value in correctness_group_metrics(rewards[:, 1].cpu().tolist(), group_size).items():
                self._metrics[mode][name].append(value)
            normalized = []
            for col in (0, 1):
                vals = grouped_standardize(rewards[:, col].cpu().tolist(), group_size)
                normalized.append(torch.tensor(vals, dtype=torch.float32, device=rewards.device))
            a_trans, a_reason = normalized
            local_size = batch["completion_ids"].shape[0]
            first = self.accelerator.process_index * local_size
            local = slice(first, first + local_size)
            a_reason = a_reason[local]
            mixed = self.tapo_alpha * a_trans[local] + (1 - self.tapo_alpha) * a_reason
            ids, active = batch["completion_ids"], batch["completion_mask"]
            segment = torch.zeros_like(active, dtype=torch.bool)
            for i in range(local_size):
                # Right padded by TRL; include generated EOS, exclude padding.
                length = int(active[i].sum().item())
                if length:
                    flags = translation_token_mask(self.processing_class, ids[i, :length].tolist())
                    segment[i, :length] = torch.tensor(flags, device=ids.device)
            batch["advantages"] = torch.where(segment, mixed[:, None], a_reason[:, None])
            batch["advantages"] *= active
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["tapo/translation_token_fraction"].append(
                (segment.sum() / active.sum().clamp(min=1)).item())
            self._tapo_rewards = None
            return batch

        def _compute_loss(self, model, inputs):
            mode = "train" if self.model.training else "eval"
            active = inputs["completion_mask"]
            advantage = inputs["advantages"]
            if advantage.shape != active.shape:
                raise RuntimeError("Token-level advantage/completion shape mismatch")
            self._metrics[mode]["tapo/token_advantage_abs"].append(
                ((advantage.abs() * active).sum() / active.sum().clamp(min=1)).item())
            if trl.__version__ == "1.9.2" and getattr(self, "tapo_log_entropy", True):
                # This version accepts B x T advantages and owns accumulation scaling.
                return super()._compute_loss(model, inputs)
            # Token-level DAPO without diagnostic entropy. Legacy TRL also needs
            # this path because its stock loss unsqueezes B x T advantages.
            ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
            mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
            scored = self._get_per_token_logps_and_entropies(
                model, ids, mask, inputs["completion_ids"].size(1), compute_entropy=False)
            logps = scored[0]
            old = inputs.get("old_per_token_logps")
            if old is None:
                old = logps.detach()
            advantage = inputs["advantages"]
            if advantage.shape != logps.shape:
                raise RuntimeError("Token-level advantage/log-probability shape mismatch")
            ratio = torch.exp(logps - old)
            clipped = ratio.clamp(1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss = -torch.minimum(ratio * advantage, clipped * advantage)
            if self.use_vllm and self.vllm_importance_sampling_correction:
                # TRL computes the sampling/training-engine correction from the
                # actual sampled log-probs; apply it in addition to PPO clipping.
                per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
            normalizer = inputs["num_items_in_batch"].clamp(min=1) / self.accelerator.num_processes
            if trl.__version__ == "1.9.2" and mode == "train":
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = (per_token_loss * inputs["completion_mask"]).sum() / normalizer
            # DAPO is already scaled for accumulation; do not divide again.
            return loss
    return TAPOTrainer
