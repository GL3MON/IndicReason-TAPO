#!/usr/bin/env python3
"""Persist MIG/GPU UUID support in vLLM 0.22; retain backups and a unified diff."""
import difflib
import importlib.util
from pathlib import Path
root = Path(importlib.util.find_spec('vllm').origin).parent
interface = root / 'platforms/interface.py'
cuda = root / 'platforms/cuda.py'
originals = {path: path.read_text() for path in (interface, cuda)}
if 'def _nvml_handle_from_device_id(' in originals[cuda]:
    print('vLLM MIG UUID patch already installed.')
    raise SystemExit(0)
source = originals[interface]
old = '            physical_device_id = device_ids[device_id]\n            return int(physical_device_id)'
new = '''            physical_device_id = device_ids[device_id].strip()
            # CUDA also accepts GPU and MIG UUIDs; preserve them for NVML lookup
            # and for child workers' CUDA_VISIBLE_DEVICES assignments.
            if physical_device_id.startswith(("GPU-", "MIG-")):
                return physical_device_id
            return int(physical_device_id)'''
assert old in source, 'Unexpected vLLM interface source; inspect before patching.'
source = source.replace(old, new)
modified = {interface: source}
source = originals[cuda]
marker = '# NVML utils\n'
helper = '''def _nvml_handle_from_device_id(device_id: int | str):
    """Resolve numeric GPU indices or CUDA-visible GPU/MIG UUIDs."""
    if isinstance(device_id, str) and device_id.startswith(("GPU-", "MIG-")):
        return pynvml.nvmlDeviceGetHandleByUUID(device_id)
    return pynvml.nvmlDeviceGetHandleByIndex(int(device_id))


'''
assert marker in source
source = source.replace(marker, helper + marker, 1)
source = source.replace('pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)', '_nvml_handle_from_device_id(physical_device_id)')
source = source.replace('pynvml.nvmlDeviceGetHandleByIndex(device_id)', '_nvml_handle_from_device_id(device_id)')
source = source.replace('handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]', '''# MIG slices cannot use inter-device NVLink collectives.
        if any(isinstance(i, str) and i.startswith("MIG-") for i in physical_device_ids):
            return False
        handles = [_nvml_handle_from_device_id(i) for i in physical_device_ids]''')
old = '            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)'
new = '''            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            except (pynvml.NVMLError_NotSupported, pynvml.NVMLError_InvalidArgument):
                # Some drivers expose capability only on the parent of a MIG slice.
                parent = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(parent)'''
assert old in source
source = source.replace(old, new, 1)
# NUMA and CPU affinity belong to the parent GPU, whereas memory must use the slice.
old = '''        handle = _nvml_handle_from_device_id(physical_device_id)

        try:
            numa_node = pynvml.nvmlDeviceGetNumaNodeId(handle)'''
new = '''        handle = _nvml_handle_from_device_id(physical_device_id)
        if isinstance(physical_device_id, str) and physical_device_id.startswith("MIG-"):
            handle = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)

        try:
            numa_node = pynvml.nvmlDeviceGetNumaNodeId(handle)'''
assert old in source
source = source.replace(old, new, 1)
modified[cuda] = source
patch = []
for path, text in modified.items():
    compile(text, str(path), 'exec')
    relative = str(path.relative_to(root))
    patch.extend(difflib.unified_diff(originals[path].splitlines(True), text.splitlines(True),
                                    fromfile='a/'+relative, tofile='b/'+relative))
for path, text in modified.items():
    backup = path.with_suffix(path.suffix+'.before-mig-patch')
    if not backup.exists(): backup.write_text(originals[path])
    path.write_text(text)
Path(__file__).with_name('vllm-0.22-mig-uuid.patch').write_text(''.join(patch))
print('Patched installed vLLM:', root)
