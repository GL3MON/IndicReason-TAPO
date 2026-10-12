#!/usr/bin/env python3
"""Expose GDN backend selection to TRL colocation via a TAPO environment option."""
import difflib
import importlib.util
from pathlib import Path


def patch_source(source):
    if 'TAPO_VLLM_GDN_PREFILL_BACKEND' in source:
        return source
    anchor = '        device_config = DeviceConfig(device=cast(Device, current_platform.device_type))'
    if anchor not in source:
        raise RuntimeError('Unexpected vLLM EngineArgs source; refusing to patch.')
    replacement = '''        # TRL 1.9.2 does not forward the optional GDN prefill backend.
        tapo_gdn_backend = os.environ.get("TAPO_VLLM_GDN_PREFILL_BACKEND")
        if tapo_gdn_backend is not None:
            if tapo_gdn_backend not in {"flashinfer", "triton", "cutedsl"}:
                raise ValueError("Invalid TAPO_VLLM_GDN_PREFILL_BACKEND")
            self.gdn_prefill_backend = tapo_gdn_backend

''' + anchor
    updated = source.replace(anchor, replacement, 1)
    if '\nimport os\n' not in updated:
        updated = updated.replace('import argparse\n', 'import argparse\nimport os\n', 1)
    compile(updated, 'arg_utils.py', 'exec')
    return updated


def main():
    path = Path(importlib.util.find_spec('vllm').origin).parent / 'engine' / 'arg_utils.py'
    original = path.read_text()
    updated = patch_source(original)
    if updated == original:
        print('TAPO GDN backend option already installed.')
        return
    backup = path.with_suffix('.py.before-tapo-gdn-patch')
    if not backup.exists():
        backup.write_text(original)
    path.write_text(updated)
    Path(__file__).with_name('vllm-0.22-tapo-gdn.patch').write_text(''.join(
        difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                             fromfile='a/engine/arg_utils.py', tofile='b/engine/arg_utils.py')))
    print('Installed TAPO GDN backend option:', path)


if __name__ == '__main__':
    main()
