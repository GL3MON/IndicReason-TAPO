"""RL training orchestration; imports ML dependencies only after supervision."""
import json
import math
from pathlib import Path
from ..constants import OPEN, CLOSE, SYSTEM, LANGUAGE_NAMES, SUPPORTED_TRL_VERSIONS
from ..io import read_jsonl
from ..formatting import text_of, translation_span, last_boxed, translation_token_mask
from ..data import normalize_question, parse_mathoctopus_prompt, prepare_rows, filter_context_rows
from ..rewards import grouped_standardize, correctness_group_metrics, make_rewards
from ..modeling import load_initial_adapter, install_vllm_sleep_policy_sync
from ..config import CONFIG_ALIASES, SMOKE_DEFAULTS, parse_args, validate_args, sampling_config_kwargs
from ..trainer import make_trainer_class


def self_test():
    # Dependency-free checks; use --smoke to exercise TRL/model forward + backward.
    assert translation_span(OPEN + "Question" + CLOSE)
    assert translation_span(OPEN + CLOSE) is None
    assert translation_span(OPEN + "x" + CLOSE + OPEN + "y" + CLOSE) is None
    assert last_boxed(r"x=2; \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{1} then \boxed{2}") == "2"
    assert last_boxed(r"\boxed{2") is None
    assert grouped_standardize([1, 1, 1, 1], 4) == [0.0] * 4
    trans = grouped_standardize([.9, .7, .5, .3], 4)
    reason = grouped_standardize([0, 1, 1, 0], 4)
    mixed = [.25 * t + .75 * r for t, r in zip(trans, reason)]
    assert mixed[0] < 0 < mixed[1] and mixed[2] > 0 > mixed[3]
    prompt = "Please answer in English.\n### Instruction:\nA question?\n### Response:"
    prompt_bn = prompt.replace("English", "Bengali").replace("A question?", "বাংলা?")
    rows = prepare_rows([{"prompt": prompt}, {"prompt": prompt_bn}],
                        [{"question": "A question?", "answer": "Steps\n#### 10"}], {"bn"})
    assert rows[0]["english_question"] == "A question?" and rows[0]["answer"] == "10"
    try:
        prepare_rows([{"prompt": prompt_bn}], [], {"bn"})
    except ValueError:
        pass
    else:
        raise AssertionError("Broken dataset order must be rejected")
    print("Self-test passed: format, boxed parsing, grouped advantages, dataset alignment.")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return

    # The supervisor imports only stdlib and starts before any ML libraries.
    # Its fresh process group contains training, vLLM and compiler workers.
    from ..runtime.memory_guard import supervise_training
    guarded_exit = supervise_training(monitor_memory=args.memory_guard)
    if guarded_exit is not None:
        return guarded_exit

    import datasets
    languages = {x.strip() for x in args.languages.split(",") if x.strip()}
    if args.data_jsonl:
        records = read_jsonl(args.data_jsonl)
        records = [r for r in records if r.get("language") in languages]
        required = {"question", "english_question", "answer", "language", "source_id"}
        if not records or any(not required.issubset(r) for r in records):
            raise ValueError("JSONL needs nonempty selected-language rows with all required fields")
    else:
        unknown = languages - set(LANGUAGE_NAMES)
        if unknown:
            raise ValueError(f"No released MathOctopus training data for {sorted(unknown)}; use --data-jsonl")
        raw = datasets.load_dataset("Mathoctopus/GSM8KInstruct_Parallel", split="train")
        original = datasets.load_dataset("openai/gsm8k", "main", split="train")
        records = prepare_rows(raw, original, languages)
    dataset = datasets.Dataset.from_list(records).shuffle(seed=args.seed)
    dataset = dataset.map(lambda row: {"prompt": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["question"]},
    ]})
    tokenizer = None
    chat_template_kwargs = {"enable_thinking": False} if "qwen3.5" in args.model.lower() else {}
    if args.use_vllm:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
        dataset = filter_context_rows(dataset, tokenizer, args.vllm_max_model_length,
                                      args.max_completion_length, chat_template_kwargs)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Prepared {len(dataset)} TRAIN rows; languages={sorted(set(dataset['language']))}")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
    if args.prepare_only:
        return

    if args.report_to == "wandb":
        import importlib.util
        import os
        if importlib.util.find_spec("wandb") is None:
            raise RuntimeError("W&B logging requires wandb: conda run -n med python -m pip install wandb")
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    import torch
    import trl
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig
    from peft import LoraConfig
    if trl.__version__ not in SUPPORTED_TRL_VERSIONS:
        raise RuntimeError(f"Unsupported TRL {trl.__version__}; expected one of {sorted(SUPPORTED_TRL_VERSIONS)}")
    if args.use_vllm and trl.__version__ != "1.9.2":
        raise RuntimeError("TAPO with vLLM requires TRL 1.9.2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for training; --self-test/--prepare-only work without one")
    set_seed(args.seed)
    if args.use_vllm and args.vllm_enforce_eager:
        import os
        os.environ["TAPO_VLLM_ENFORCE_EAGER"] = "1"
    if args.use_vllm and args.vllm_gdn_prefill_backend != "auto":
        import os
        os.environ["TAPO_VLLM_GDN_PREFILL_BACKEND"] = args.vllm_gdn_prefill_backend
    bf16 = torch.cuda.is_bf16_supported() if args.precision == "auto" else args.precision == "bf16"
    if bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 precision is not supported by this GPU")
    fp16 = args.precision == "fp16" or (args.precision == "auto" and not bf16)
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # No device_map='auto': Accelerator owns placement, including distributed training.
    model_config = AutoConfig.from_pretrained(args.model)
    model_class = AutoModelForCausalLM
    if model_config.model_type == "qwen3_5":
        # Keep original parameter names for vLLM synchronization; train text LoRA only.
        from transformers import Qwen3_5ForConditionalGeneration
        model_class = Qwen3_5ForConditionalGeneration
    model = model_class.from_pretrained(args.model, config=model_config, dtype=dtype,
                                        attn_implementation="sdpa")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
        model.config.text_config.pad_token_id = tokenizer.pad_token_id
    if args.init_adapter is not None:
        model = load_initial_adapter(model, args.init_adapter)
        print(f"Initialized RL from trainable SFT adapter: {args.init_adapter}", flush=True)
    # Full fp16 trainable parameters conflict with GradScaler; keep FP32 weights on T4.
    if args.full_finetune and fp16:
        model = model.float()
    legacy_config = {"max_prompt_length": 1024, "use_liger_loss": False} if trl.__version__ == "0.24.0" else {}
    vllm_config = {} if trl.__version__ == "0.24.0" else dict(
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_importance_sampling_correction=args.vllm_importance_sampling_correction,
        vllm_importance_sampling_mode=args.vllm_importance_sampling_mode,
        vllm_importance_sampling_clip_max=args.vllm_importance_sampling_clip_max,
    )
    config = GRPOConfig(
        **legacy_config,
        **vllm_config,
        **sampling_config_kwargs(args, trl.__version__),
        **({"chat_template_kwargs": chat_template_kwargs} if trl.__version__ == "1.9.2" else {}),
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.group_size,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=0.0,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        loss_type="dapo",
        importance_sampling_level="token",
        num_iterations=args.num_iterations,
        use_vllm=args.use_vllm,
        mask_truncated_completions=args.mask_truncated_completions,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=fp16,
        optim=args.optim,
        logging_steps=args.logging_steps,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    peft_config = None if args.full_finetune or args.init_adapter is not None else LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none", task_type="CAUSAL_LM",
    )
    trainer = make_trainer_class()(
        model=model,
        processing_class=tokenizer,
        reward_funcs=list(make_rewards()),
        train_dataset=dataset,
        args=config,
        peft_config=peft_config,
        tapo_alpha=args.alpha,
        tapo_log_entropy=args.log_entropy,
    )
    print("TAPO enabled: separate token advantages; English translation + English reasoning.")
    result = trainer.train(resume_from_checkpoint=args.resume)
    if args.smoke:
        if trainer.state.global_step != args.max_steps or not math.isfinite(result.training_loss):
            raise RuntimeError("Smoke test did not finish the requested steps with finite loss")
        print(f"Smoke test completed: {trainer.state.global_step} optimizer steps; metrics={result.metrics}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved {'full model' if args.full_finetune else 'LoRA adapter'} to {args.output_dir}")
    if args.smoke:
        print("Smoke test checks execution only; short completions may truncate outputs/produce zero rewards.")
