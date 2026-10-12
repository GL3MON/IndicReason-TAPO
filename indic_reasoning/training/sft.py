"""Completion-only supervised training with validation-selected LoRA checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from ..io import read_jsonl
from ..config import parse_args as parse_rl_args
from ..sft_data import prepare_sft_rows, split_sft_rows, tokenize_sft_row

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-config", type=Path, default=Path("configs/tapo_ta_qwen35.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen35_ta_sft"))
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--exclude-validation-examples", type=Path, help="Keep these previously trained source IDs out of validation")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, help="Train this many full epochs instead of --max-steps")
    parser.add_argument("--save-steps", type=int, default=32)
    parser.add_argument("--select-best", action="store_true", help="Select the checkpoint with lowest held-out TRAIN validation loss")
    parser.add_argument("--run-name", default="qwen35-tamil-sft-warmup")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--memory-guard", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.max_samples, args.eval_samples, args.max_steps, args.batch_size,
           args.grad_accum, args.max_length) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("SFT counts, lengths and learning rate must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.save_steps < 1:
        parser.error("--save-steps must be positive")
    return args


def main():
    args = parse_args()
    from ..runtime.memory_guard import supervise_training
    exit_code = supervise_training(monitor_memory=args.memory_guard)
    if exit_code is not None:
        return exit_code
    import os
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    base = parse_rl_args(["--config", str(args.rl_config)])
    if base.model != "Qwen/Qwen3.5-2B" or base.full_finetune or base.init_adapter:
        raise ValueError("This warm-up expects fresh Qwen3.5-2B LoRA with no initial adapter")
    records = read_jsonl(base.data_jsonl)
    cached = sorted((Path.home() / ".cache/huggingface/datasets/openai___gsm8k/main").glob("*/*/gsm8k-train.arrow"))
    if cached:
        official_path = max(cached, key=lambda p: p.stat().st_mtime)
        official = Dataset.from_file(str(official_path))
        provenance = str(official_path)
    else:
        official = load_dataset("openai/gsm8k", "main", split="train")
        provenance = "openai/gsm8k main TRAIN"
    rows = prepare_sft_rows(records, official)
    excluded_eval_ids = {json.loads(line)["source_id"] for line in args.exclude_validation_examples.read_text().splitlines() if line.strip()} if args.exclude_validation_examples else set()
    selected_train, selected_eval = split_sft_rows(rows, args.max_samples, args.eval_samples, args.seed, excluded_eval_ids)
    tokenizer = AutoTokenizer.from_pretrained(base.model, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, train_tokens, eval_rows, eval_tokens = [], [], [], []
    for selected, kept, tokens in ((selected_train, train_rows, train_tokens),
                                   (selected_eval, eval_rows, eval_tokens)):
        for row in selected:
            encoded = tokenize_sft_row(row, tokenizer, args.max_length)
            if encoded is not None:
                kept.append(row)
                tokens.append(encoded)
    if not train_tokens or not eval_tokens:
        raise ValueError("No complete SFT examples fit the token budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train_examples", train_rows), ("eval_examples", eval_rows)):
        (args.output_dir / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in subset))
    dataset_report = dict(source="openai/gsm8k main TRAIN", source_cache=provenance,
                          matched_train_questions=len(rows), train_examples=len(train_rows),
                          eval_examples=len(eval_rows), train_source_ids=[r["source_id"] for r in train_rows],
                          eval_source_ids=[r["source_id"] for r in eval_rows], max_length=args.max_length,
                          max_sequence_tokens=max(len(r["input_ids"]) for r in train_tokens + eval_tokens),
                          completion_only_loss=True, test_questions_used=0, seed=args.seed,
                          validation_excluded_examples=str(args.exclude_validation_examples) if args.exclude_validation_examples else None)
    (args.output_dir / "dataset_report.json").write_text(json.dumps(dataset_report, indent=2) + "\n")
    print(f"Matched {len(rows)} official TRAIN questions; SFT train={len(train_rows)}, eval={len(eval_rows)}; "
          f"max sequence={dataset_report['max_sequence_tokens']} tokens. Gold targets retained in full.", flush=True)
    if args.prepare_only:
        return
    eval_ids = {r["source_id"] for r in selected_eval}
    (args.output_dir / "rl_train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in records if r["source_id"] not in eval_ids))
    adapter_dir = args.output_dir / "adapter"
    if (adapter_dir / "adapter_model.safetensors").exists():
        raise ValueError("SFT adapter already exists; choose a fresh --output-dir")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoConfig, Qwen3_5ForConditionalGeneration, Trainer,
                              TrainingArguments, DataCollatorForSeq2Seq, set_seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This Qwen3.5 SFT run requires a BF16 CUDA GPU")
    set_seed(args.seed)
    os.environ["WANDB_PROJECT"] = base.wandb_project
    os.environ["WANDB_MODE"] = base.wandb_mode
    if base.wandb_entity:
        os.environ["WANDB_ENTITY"] = base.wandb_entity
    config = AutoConfig.from_pretrained(base.model)
    config.use_cache = False
    config.text_config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    config.text_config.pad_token_id = tokenizer.pad_token_id
    model = Qwen3_5ForConditionalGeneration.from_pretrained(base.model, config=config,
                                                           dtype=torch.bfloat16, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(r=base.lora_r, lora_alpha=base.lora_alpha,
                            lora_dropout=base.lora_dropout, target_modules=base.lora_target_modules,
                            bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    training_args = TrainingArguments(output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, max_steps=-1 if args.epochs else args.max_steps,
        num_train_epochs=args.epochs or 1,
        learning_rate=args.lr, warmup_ratio=0.05, lr_scheduler_type="linear",
        max_grad_norm=1.0, bf16=True, fp16=False, optim="adamw_torch_fused",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=4, save_steps=args.save_steps, save_total_limit=2, report_to=args.report_to,
        eval_strategy="steps" if args.select_best else "no", eval_steps=args.save_steps,
        load_best_model_at_end=args.select_best, metric_for_best_model="eval_loss", greater_is_better=False,
        run_name=args.run_name, seed=args.seed, dataloader_num_workers=0,
        prediction_loss_only=True)
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=Dataset.from_list(train_tokens), eval_dataset=Dataset.from_list(eval_tokens),
                      processing_class=tokenizer,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8,
                                                          label_pad_token_id=-100))
    before = trainer.evaluate(metric_key_prefix="before_sft")
    result = trainer.train()
    after = trainer.evaluate(metric_key_prefix="after_sft")
    expected_steps = math.ceil(len(train_tokens) / (args.batch_size * args.grad_accum)) * args.epochs if args.epochs else args.max_steps
    if trainer.state.global_step != expected_steps or not math.isfinite(result.training_loss):
        raise RuntimeError("SFT did not complete its requested steps with finite loss")
    if not math.isfinite(after["after_sft_loss"]):
        raise RuntimeError("SFT validation loss is not finite")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    report = dict(status="complete", model=base.model, adapter_dir=str(adapter_dir),
                  global_step=trainer.state.global_step, learning_rate=args.lr,
                  epochs=args.epochs, best_checkpoint=trainer.state.best_model_checkpoint,
                  best_validation_loss=trainer.state.best_metric, selection_metric="eval_loss" if args.select_best else None,
                  dataset=dataset_report, before_sft=before, after_sft=after,
                  training_metrics=result.metrics,
                  adapter_sha256=hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest())
    (args.output_dir / "sft_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"SFT completed: {trainer.state.global_step} steps; held-out TRAIN-subset loss "
          f"{before['before_sft_loss']:.4f} -> {after['after_sft_loss']:.4f}; adapter={adapter_dir}", flush=True)
