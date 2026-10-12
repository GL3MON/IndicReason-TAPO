"""TAPO reward, token segmentation, and policy-gradient regression checks."""
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch
from trl import GRPOTrainer

from train_tapo_trl import (
    CLOSE, OPEN, grouped_standardize, make_rewards, make_trainer_class,
    translation_token_mask,
)


class CharacterTokenizer:
    def __init__(self, round_trip=True):
        self.round_trip = round_trip

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)

    def __call__(self, text, **kwargs):
        return {
            'input_ids': [ord(c) for c in text] if self.round_trip else [],
            'offset_mapping': [(i, i + 1) for i in range(len(text))],
        }


class TAPOTests(unittest.TestCase):
    def test_reward_requires_answer_after_translation(self):
        translation, correctness = make_rewards()
        question = 'What is two plus two?'
        good = OPEN + question + CLOSE + r' Two plus two is four. \boxed{4}'
        inside = OPEN + question + r' \boxed{4}' + CLOSE
        malformed = OPEN + question + r' \boxed{4}'
        self.assertAlmostEqual(translation([good], [question])[0], 1.0)
        self.assertEqual(correctness([good, inside, malformed], ['4'] * 3), [1.0, 0.0, 0.0])
        equivalent = OPEN + question + CLOSE + r' \boxed{\frac{8}{2}}'
        self.assertEqual(correctness([equivalent], ['4']), [1.0])

    def test_actual_token_boundary_and_fallback(self):
        text = 'Preamble ' + OPEN + 'Question' + CLOSE + r' Reasoning \boxed{4}'
        ids = [ord(c) for c in text]
        boundary = text.index(CLOSE) + len(CLOSE)
        expected = [i < boundary for i in range(len(ids))]
        for round_trip in (True, False):
            self.assertEqual(translation_token_mask(CharacterTokenizer(round_trip), ids), expected)
        self.assertEqual(translation_token_mask(CharacterTokenizer(), [ord(c) for c in 'No tags']), [False] * 7)

    def test_separate_advantages_and_nonzero_policy_gradient(self):
        cls = make_trainer_class()
        trainer = object.__new__(cls)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.tapo_alpha = .25
        trainer.num_generations = 2
        trainer.processing_class = CharacterTokenizer()
        trainer.accelerator = SimpleNamespace(
            process_index=0, num_processes=1,
            reduce=lambda x, **kwargs: x,
            gather=lambda x: x,
        )
        trainer._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        text = OPEN + 'Q' + CLOSE + ' Reason'
        ids = torch.tensor([[ord(c) for c in text]] * 2)
        active = torch.ones_like(ids)
        active[:, -1] = 0  # padding must receive no advantage or gradient

        def generated(instance, inputs):
            instance._tapo_rewards = torch.tensor([[1., 0.], [0., 1.]])
            return {'completion_ids': ids, 'completion_mask': active}

        with patch.object(GRPOTrainer, '_generate_and_score_completions', generated):
            batch = trainer._generate_and_score_completions([])
        trans = torch.tensor(grouped_standardize([1., 0.], 2))
        reason = torch.tensor(grouped_standardize([0., 1.], 2))
        boundary = text.index(CLOSE) + len(CLOSE)
        torch.testing.assert_close(batch['advantages'][:, 0], .25 * trans + .75 * reason)
        torch.testing.assert_close(batch['advantages'][:, boundary], reason)
        self.assertTrue(torch.equal(batch['advantages'][:, -1], torch.zeros(2)))

        # Exercise the installed TRL DAPO loss with the resulting B x T advantages.
        logps = torch.zeros_like(active, dtype=torch.float32, requires_grad=True)
        trainer._get_per_token_logps_and_entropies = lambda *args, **kwargs: (
            logps, torch.ones_like(logps), None)
        trainer.aux_loss_enabled = False
        trainer.top_entropy_quantile = 1.
        trainer.off_policy_mask_threshold = None
        trainer.importance_sampling_level = 'token'
        trainer.beta = 0.
        trainer.loss_type = 'dapo'
        trainer.epsilon_low = .2
        trainer.epsilon_high = .28
        trainer.use_vllm = False
        trainer._entropy_bonus_enabled = False
        trainer.current_gradient_accumulation_steps = 2
        trainer.args = SimpleNamespace(delta=None, steps_per_generation=2)
        batch.update(prompt_ids=torch.ones((2, 1), dtype=torch.long),
                     prompt_mask=torch.ones((2, 1), dtype=torch.long),
                     num_items_in_batch=active.sum())
        loss = trainer._compute_loss(trainer.model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        expected = -batch['advantages'] * active / active.sum()
        torch.testing.assert_close(logps.grad, expected)
        self.assertGreater(logps.grad.abs().sum().item(), 0.)
        # Fast loss must preserve both values and gradients from stock DAPO.
        stock_loss = loss.detach().clone()
        logps.grad = None
        trainer.tapo_log_entropy = False
        fast_loss = trainer._compute_loss(trainer.model, batch)
        fast_loss.backward()
        torch.testing.assert_close(fast_loss.detach(), stock_loss)
        torch.testing.assert_close(logps.grad, expected)
        # Reused rollouts must retain PPO clipping and accumulation scaling too.
        batch['old_per_token_logps'] = torch.zeros_like(logps)
        with torch.no_grad():
            logps[:, :boundary] = -.9
            logps[:, boundary:] = .9
        trainer.current_gradient_accumulation_steps = 1
        results = []
        for log_entropy in (True, False):
            trainer.tapo_log_entropy = log_entropy
            logps.grad = None
            clipped_loss = trainer._compute_loss(trainer.model, batch)
            clipped_loss.backward()
            results.append((clipped_loss.detach().clone(), logps.grad.clone()))
        torch.testing.assert_close(results[0][0], results[1][0])
        torch.testing.assert_close(results[0][1], results[1][1])
        # vLLM corrections must also reach token-level TAPO advantages.
        trainer.use_vllm = True
        trainer.vllm_importance_sampling_correction = True
        batch['importance_sampling_ratio'] = torch.tensor([[.5], [1.5]])
        results = []
        for log_entropy in (True, False):
            trainer.tapo_log_entropy = log_entropy
            logps.grad = None
            corrected_loss = trainer._compute_loss(trainer.model, batch)
            corrected_loss.backward()
            results.append((corrected_loss.detach().clone(), logps.grad.clone()))
        torch.testing.assert_close(results[0][0], results[1][0])
        torch.testing.assert_close(results[0][1], results[1][1])


if __name__ == '__main__':
    unittest.main()
