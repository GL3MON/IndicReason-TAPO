"""Trainable adapter loading and safe vLLM sleep-mode weight synchronization."""

def load_initial_adapter(model, path):
    """Continue the trained SFT adapter rather than initializing new LoRA weights."""
    from peft import PeftConfig, PeftModel
    config = PeftConfig.from_pretrained(str(path))
    if config.base_model_name_or_path != model.name_or_path:
        raise ValueError(f"SFT adapter base {config.base_model_name_or_path!r} does not match {model.name_or_path!r}")
    return PeftModel.from_pretrained(model, str(path), is_trainable=True)


def install_vllm_sleep_policy_sync(generation):
    """Restore the current policy directly after waking sleeping vLLM weights.

    TRL 1.9.2 requests a disk checkpoint reload in generate(). Its layerwise
    reload materializes temporary tensors before copying into the existing
    storage, increasing peak VRAM and triggering a MIG NVML assertion on OOM.
    Our unquantized policy sync already writes every model parameter, including
    merged LoRA weights. Use it instead of allocating/reloading the base model.
    """
    sync_policy = generation.sync_weights
    collective_rpc = generation.llm.collective_rpc

    def defer_sync():
        # Synchronize at generation time even when GRPO caches the outer sync.
        return None

    def sync_instead_of_reload(method, *args, **kwargs):
        if method != "reload_weights":
            return collective_rpc(method, *args, **kwargs)
        if args or kwargs:
            raise ValueError("TAPO replaces only TRL's default checkpoint reload")
        # TRL's original sync wakes the sleep-aware weight pool, copies all
        # parameters into its existing tensors, then resets the prefix cache.
        # Propagate any failure so generation cannot use stale/discarded weights.
        sync_policy()
        return None

    generation.sync_weights = defer_sync
    generation.llm.collective_rpc = sync_instead_of_reload
