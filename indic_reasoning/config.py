"""YAML and CLI configuration, validation, and backend sampling controls."""
import argparse
import math
from pathlib import Path

CONFIG_ALIASES = {
    "per_device_train_batch_size": "batch_size",
    "num_generations": "group_size",
    "gradient_accumulation_steps": "grad_accum",
    "learning_rate": "lr",
    "num_train_epochs": "epochs",
    "tapo_alpha": "alpha",
}


SMOKE_DEFAULTS = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct", "max_samples": 8,
    "max_steps": 2, "batch_size": 1, "group_size": 2, "grad_accum": 2,
    "max_completion_length": 256, "output_dir": "outputs/tapo_smoke_test",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML config; explicit CLI flags override YAML")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--languages", default="bn")
    parser.add_argument("--data-jsonl", type=Path)
    parser.add_argument("--output-dir", default="tapo-qwen25-indic")
    parser.add_argument("--max-samples", type=int, default=2000, help="0 = all selected training rows")
    parser.add_argument("--max-steps", type=int, default=100, help="-1 = use epochs")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", "--per-device-train-batch-size", type=int, default=1,
                        help="Completion sequences per device per training microbatch")
    parser.add_argument("--group-size", "--num-generations", type=int, default=8,
                        help="Rollouts per unique prompt")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=.25)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--full-finetune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", default=None, help="Checkpoint directory to resume")
    parser.add_argument("--init-adapter", type=Path, default=None,
                        help="Initialize trainable LoRA weights from an SFT adapter; fresh RL optimizer")
    parser.add_argument("--prepare-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--self-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-p", type=float, default=1.)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.)
    parser.add_argument("--presence-penalty", type=float, default=0.)
    parser.add_argument("--repetition-penalty", type=float, default=1.)
    parser.add_argument("--epsilon", type=float, default=.2)
    parser.add_argument("--epsilon-high", type=float, default=.28)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--mask-truncated-completions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--memory-guard", action=argparse.BooleanOptionalAction, default=True,
                        help="Monitor container RAM; --no-memory-guard keeps CPU limits and worker cleanup")
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False,
                        help="Disable vLLM compilation/CUDA graphs; requires patches/apply_vllm_eager.py")
    parser.add_argument("--vllm-gdn-prefill-backend", choices=["auto", "flashinfer", "triton", "cutedsl"],
                        default="auto", help="GDN backend; requires patches/apply_vllm_gdn_backend.py")
    parser.add_argument("--vllm-mode", choices=["colocate"], default="colocate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=.4)
    parser.add_argument("--vllm-enable-sleep-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-max-model-length", type=int, default=2048)
    parser.add_argument("--vllm-importance-sampling-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-importance-sampling-mode",
                        choices=["sequence_mask", "sequence_truncate", "token_mask", "token_truncate"],
                        default="token_truncate")
    parser.add_argument("--vllm-importance-sampling-clip-max", type=float, default=3.)
    parser.add_argument("--log-entropy", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute diagnostic entropy/stock loss metrics; disable to reduce memory and work")
    parser.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=False,
                        help="Log prompts, completions and rewards to W&B at logging_steps intervals")
    parser.add_argument("--num-completions-to-print", type=int, default=0,
                        help="Completions to print in the terminal; 0 keeps completion logs in W&B/files only")
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb-project", default="indic-reasoning-tapo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--warmup-ratio", type=float, default=0.)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.)
    parser.add_argument("--lora-target-modules", default="all-linear")

    # Discover the file without parsing the remaining flags or importing training dependencies.
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    preliminary.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None)
    selected, _ = preliminary.parse_known_args(argv)
    defaults = {}
    if selected.config:
        try:
            import yaml
            loaded = yaml.safe_load(selected.config.read_text(encoding="utf-8"))
        except ImportError:
            parser.error("YAML configuration requires PyYAML: pip install pyyaml")
        except (OSError, yaml.YAMLError) as exc:
            parser.error(f"Cannot load config {selected.config}: {exc}")
        if not isinstance(loaded, dict):
            parser.error("YAML config must be a mapping of argument names to values")
        actions = {a.dest: a for a in parser._actions if a.dest not in {"help", "config"}}
        for key, value in loaded.items():
            name = CONFIG_ALIASES.get(key, key)
            if name not in actions:
                parser.error(f"Unknown YAML argument: {key}")
            if name in defaults:
                parser.error(f"Duplicate YAML arguments for {name}")
            action = actions[name]
            if isinstance(action, argparse.BooleanOptionalAction):
                if not isinstance(value, bool):
                    parser.error(f"{key} must be a YAML boolean (true/false)")
            elif value is None:
                if name not in {"data_jsonl", "resume", "init_adapter", "wandb_entity", "run_name"}:
                    parser.error(f"{key} cannot be null")
            elif name in {"languages", "lora_target_modules"} and isinstance(value, list):
                if not value or any(not isinstance(item, str) for item in value):
                    parser.error(f"{key} must contain strings")
                if name == "languages":
                    value = ",".join(value)
            elif action.type in {int, float}:
                if isinstance(value, bool) or (action.type is int and not isinstance(value, int)):
                    parser.error(f"{key} must be a {'whole number' if action.type is int else 'number'}")
                try:
                    value = action.type(value)
                except (ValueError, TypeError):
                    parser.error(f"Invalid value for {key}: {value}")
                if not math.isfinite(value):
                    parser.error(f"{key} must be finite")
            elif action.type is Path:
                if not isinstance(value, str):
                    parser.error(f"{key} must be a path string")
                value = Path(value)
            elif not isinstance(value, str):
                parser.error(f"{key} must be a string")
            if action.choices and value not in action.choices:
                parser.error(f"Invalid {key}: choose from {action.choices}")
            defaults[name] = value
    smoke = selected.smoke if selected.smoke is not None else defaults.get("smoke", False)
    if smoke:
        parser.set_defaults(**SMOKE_DEFAULTS)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args


def validate_args(args, parser):
    for name in ("batch_size", "grad_accum", "max_completion_length", "num_iterations",
                 "logging_steps", "save_steps", "save_total_limit", "lora_r", "lora_alpha"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not 0 <= args.alpha <= 1 or args.group_size < 2 or args.max_samples < 0:
        parser.error("Require alpha in [0,1], group-size >= 2, max-samples >= 0")
    if args.max_steps != -1 and args.max_steps < 1:
        parser.error("max_steps must be -1 (epochs) or positive")
    if (args.epochs <= 0 or args.lr <= 0 or args.temperature <= 0 or not 0 < args.top_p <= 1
            or args.top_k < 0 or args.epsilon < 0 or args.epsilon_high < 0
            or not 0 <= args.warmup_ratio <= 1 or not 0 <= args.lora_dropout < 1
            or args.weight_decay < 0 or args.max_grad_norm < 0 or args.dataloader_num_workers < 0
            or args.num_completions_to_print < 0):
        parser.error("Invalid training or sampling parameter range")
    if (not all(math.isfinite(v) for v in (args.min_p, args.presence_penalty, args.repetition_penalty))
            or not 0 <= args.min_p <= 1 or not -2 <= args.presence_penalty <= 2
            or args.repetition_penalty <= 0):
        parser.error("Require min_p in [0,1], presence_penalty in [-2,2], repetition_penalty > 0")
    if args.presence_penalty != 0 and not args.use_vllm:
        parser.error("presence_penalty requires --use-vllm")
    if args.use_vllm:
        if not 0 < args.vllm_gpu_memory_utilization < 1:
            parser.error("vllm_gpu_memory_utilization must be between 0 and 1")
        if args.vllm_max_model_length <= args.max_completion_length:
            parser.error("vllm_max_model_length must exceed max_completion_length to leave room for the prompt")
        if args.vllm_importance_sampling_clip_max <= 0:
            parser.error("vllm_importance_sampling_clip_max must be positive")
    if not args.languages.strip():
        parser.error("Select at least one language")
    if args.init_adapter is not None and args.full_finetune:
        parser.error("init_adapter requires LoRA training, not full_finetune")
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    effective_batch = args.batch_size * args.grad_accum * world_size
    if effective_batch % args.group_size:
        parser.error(f"batch_size * grad_accum * world_size ({effective_batch}) must be divisible "
                     f"by group_size ({args.group_size})")


def sampling_config_kwargs(args, trl_version):
    """Forward sampling controls to TRL's actual generation backend."""
    if trl_version == "0.24.0":
        if args.min_p != 0 or args.repetition_penalty != 1 or args.presence_penalty != 0:
            raise ValueError("These additional sampling controls require TRL 1.9.2")
        return {}
    return dict(
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        generation_kwargs={"presence_penalty": args.presence_penalty} if args.presence_penalty else None,
    )
