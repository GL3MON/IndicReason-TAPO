import json
from pathlib import Path
import tempfile
import unittest
from indic_reasoning.checkpoints import latest_checkpoint,snapshot_checkpoint

class SnapshotTests(unittest.TestCase):
    def checkpoint(self,root,step,complete=True):
        p=root/f'checkpoint-{step}';p.mkdir()
        (p/'adapter_model.safetensors').write_bytes(b'weights'+str(step).encode())
        (p/'adapter_config.json').write_text('{}')
        if complete:(p/'trainer_state.json').write_text(json.dumps({'global_step':step}))
        return p
    def test_skips_partial_save_and_preserves_rotated_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);saved=self.checkpoint(root,25);self.checkpoint(root,50,False)
            self.assertEqual(latest_checkpoint(root),saved)
            frozen=root/'snapshot';meta=snapshot_checkpoint(saved,frozen)
            self.assertEqual(meta['global_step'],25)
            self.assertEqual((frozen/'adapter_model.safetensors').read_bytes(),(saved/'adapter_model.safetensors').read_bytes())
            other=self.checkpoint(root,75)
            with self.assertRaises(ValueError):snapshot_checkpoint(other,frozen)

if __name__=='__main__':unittest.main()
