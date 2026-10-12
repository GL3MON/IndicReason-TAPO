#!/usr/bin/env python3
"""Opt-in eager execution for TAPO colocated vLLM via TAPO_VLLM_ENFORCE_EAGER."""
import difflib
import importlib.util
from pathlib import Path
root=Path(importlib.util.find_spec('vllm').origin).parent
path=root/'engine/arg_utils.py';old=path.read_text()
if 'TAPO_VLLM_ENFORCE_EAGER' in old:
    print('vLLM TAPO eager option already installed.');raise SystemExit(0)
anchor='        current_platform.pre_register_and_update()\n\n        device_config ='
replacement='''        current_platform.pre_register_and_update()
        # Explicit opt-in for small MIG slices used by TAPO/TRL colocation.
        # TRL 1.9.2 does not expose the LLM enforce_eager constructor argument.
        if os.environ.get("TAPO_VLLM_ENFORCE_EAGER") == "1":
            self.enforce_eager = True

        device_config ='''
assert anchor in old, 'Unexpected vLLM engine config source.'
new=old.replace(anchor,replacement,1).replace('import argparse\n', 'import argparse\nimport os\n', 1);compile(new,str(path),'exec')
backup=path.with_suffix('.py.before-tapo-eager-patch')
if not backup.exists():backup.write_text(old)
path.write_text(new)
Path(__file__).with_name('vllm-0.22-tapo-eager.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/engine/arg_utils.py',tofile='b/engine/arg_utils.py')))
print('Installed TAPO opt-in eager execution:',path)
