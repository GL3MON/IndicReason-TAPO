"""Configuration precedence and GRPO batch validation without model imports."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from train_tapo_trl import parse_args, install_vllm_sleep_policy_sync


class ConfigTests(unittest.TestCase):
    def parse_yaml(self, text, *flags):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(text)
            with patch.dict(os.environ, {'WORLD_SIZE': '1'}):
                return parse_args(['--config', str(path), *flags])

    def assert_invalid(self, text, error):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.parse_yaml(text)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(error, stderr.getvalue())

    def test_yaml_types_aliases_and_cli_precedence(self):
        args = self.parse_yaml('''
per_device_train_batch_size: 2
num_generations: 4
gradient_accumulation_steps: 2
learning_rate: 2.0e-6
num_train_epochs: 3
languages: [bn, ja]
data_jsonl: data/train.jsonl
full_finetune: true
lora_target_modules: [q_proj, v_proj]
''', '--lr', '5e-6', '--no-full-finetune')
        self.assertEqual((args.batch_size, args.group_size, args.grad_accum), (2, 4, 2))
        self.assertEqual(args.lr, 5e-6)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.languages, 'bn,ja')
        self.assertEqual(args.data_jsonl, Path('data/train.jsonl'))
        self.assertFalse(args.full_finetune)
        self.assertEqual(args.lora_target_modules, ['q_proj', 'v_proj'])

    def test_smoke_defaults_do_not_overwrite_explicit_values(self):
        args = self.parse_yaml('''smoke: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
max_completion_length: 128
''', '--max-steps', '3')
        self.assertEqual(args.model, 'Qwen/Qwen2.5-0.5B-Instruct')
        self.assertEqual((args.batch_size, args.grad_accum, args.group_size), (2, 1, 2))
        self.assertEqual(args.max_completion_length, 128)
        self.assertEqual(args.max_steps, 3)

    def test_rejects_misspelled_duplicate_and_unsafe_yaml(self):
        self.assert_invalid('num_rollots: 4', 'Unknown YAML argument')
        self.assert_invalid('group_size: 4\nnum_generations: 4', 'Duplicate YAML arguments')
        self.assert_invalid('gradient_checkpointing: "false"', 'YAML boolean')
        self.assert_invalid('per_device_train_batch_size: 1.5', 'whole number')
        self.assert_invalid('num_generations: 0', 'group-size >= 2')
        self.assert_invalid('learning_rate: .nan', 'must be finite')
        self.assert_invalid('model: null', 'cannot be null')
        self.assert_invalid('- bn', 'must be a mapping')
        self.assert_invalid('model: !!python/object:builtins.object {}', 'Cannot load config')

    def test_batch_validation_counts_microbatch_and_world_size(self):
        self.assert_invalid('batch_size: 3\ngrad_accum: 2\ngroup_size: 4', 'must be divisible')
        with patch.dict(os.environ, {'WORLD_SIZE': '2'}):
            args = parse_args(['--batch-size', '2', '--grad-accum', '1', '--group-size', '4'])
        self.assertEqual(args.batch_size, 2)

    def test_fast_profile_batch_is_valid(self):
        with patch.dict(os.environ, {'WORLD_SIZE': '1'}):
            normal = parse_args(['--config', 'configs/tapo_ta_qwen35.yaml'])
            fast = parse_args(['--config', 'configs/tapo_ta_qwen35_sft_2000_rl.yaml'])
        self.assertEqual((fast.batch_size * fast.grad_accum) % fast.group_size, 0)
        for name in ('model', 'group_size', 'max_samples', 'epochs', 'lr'):
            self.assertEqual(getattr(normal, name), getattr(fast, name))
        self.assertEqual(fast.optim, 'adamw_torch_fused')
        self.assertGreater(fast.max_completion_length, 0)
        self.assertFalse(fast.log_entropy)

    def test_wandb_yaml_and_cli_disable(self):
        args = self.parse_yaml("report_to: wandb\nwandb_project: test-tapo\nwandb_entity: null\nrun_name: bengali\nwandb_mode: offline")
        self.assertEqual(args.report_to, "wandb")
        self.assertEqual(args.wandb_project, "test-tapo")
        self.assertEqual(args.run_name, "bengali")
        self.assertIsNone(args.wandb_entity)
        self.assertEqual(args.wandb_mode, "offline")
        args = self.parse_yaml("report_to: wandb", "--report-to", "none")
        self.assertEqual(args.report_to, "none")

    def test_vllm_config_and_memory_validation(self):
        with patch.dict(os.environ, {'WORLD_SIZE': '1'}):
            args = parse_args(['--config', 'configs/tapo_ta_qwen35_sft_2000_rl.yaml'])
        self.assertTrue(args.use_vllm)
        self.assertTrue(args.vllm_enable_sleep_mode)
        self.assertEqual(args.vllm_mode, 'colocate')
        self.assertEqual((args.batch_size * args.grad_accum) % args.group_size, 0)
        self.assertEqual(args.vllm_gpu_memory_utilization, .4)
        self.assert_invalid('use_vllm: true\nvllm_gpu_memory_utilization: 1.2', 'must be between')
        self.assert_invalid('use_vllm: true\nvllm_max_model_length: 64', 'leave room for the prompt')

    def test_sleep_reload_preserves_latest_policy(self):
        from types import SimpleNamespace
        events = []
        def rpc(method, *args, **kwargs):
            events.append(method)
            return "loaded"
        generation = SimpleNamespace(sync_weights=lambda: events.append("policy_sync"),
                                     llm=SimpleNamespace(collective_rpc=rpc))
        install_vllm_sleep_policy_sync(generation)
        generation.sync_weights()
        self.assertEqual(events, [])
        self.assertIsNone(generation.llm.collective_rpc("reload_weights"))
        self.assertEqual(events, ["policy_sync"])
        self.assertEqual(generation.llm.collective_rpc("other_rpc"), "loaded")
        self.assertEqual(events[-1], "other_rpc")
        # Sync again after every reload even if the outer trainer cached its sync.
        generation.llm.collective_rpc("reload_weights")
        self.assertEqual(events, ["policy_sync", "other_rpc", "policy_sync"])

    def test_sleep_sync_failure_is_not_swallowed(self):
        from types import SimpleNamespace
        def failed_sync():
            raise RuntimeError("policy sync failed")
        generation = SimpleNamespace(sync_weights=failed_sync,
                                     llm=SimpleNamespace(collective_rpc=lambda *args: self.fail("disk reload called")))
        install_vllm_sleep_policy_sync(generation)
        with self.assertRaisesRegex(RuntimeError, "policy sync failed"):
            generation.llm.collective_rpc("reload_weights")
        with self.assertRaisesRegex(ValueError, "default checkpoint reload"):
            generation.llm.collective_rpc("reload_weights", weights_path="other-model")

    def test_qwen35_recommended_sampling_and_validation(self):
        from train_tapo_trl import sampling_config_kwargs
        args = parse_args(['--config', 'configs/tapo_ta_qwen35.yaml'])
        self.assertEqual((args.temperature, args.top_p, args.top_k, args.min_p,
                          args.presence_penalty, args.repetition_penalty),
                         (1.0, 1.0, 20, 0.0, 2.0, 1.0))
        forwarded = sampling_config_kwargs(args, '1.9.2')
        self.assertEqual(forwarded['generation_kwargs'], {'presence_penalty': 2.0})
        for flag, value in (('--min-p', '1.1'), ('--presence-penalty', '3'),
                            ('--repetition-penalty', '0'), ('--min-p', 'nan')):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--config', 'configs/tapo_ta_qwen35.yaml', flag, value])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--presence-penalty', '2'])
        defaults = parse_args([])
        self.assertEqual(sampling_config_kwargs(defaults, '0.24.0'), {})

    def test_shipped_configs(self):
        with patch.dict(os.environ, {'WORLD_SIZE': '1'}):
            full = parse_args(['--config', 'configs/tapo_ta_qwen35.yaml'])
            smoke = parse_args(['--smoke', '--batch-size', '2', '--grad-accum', '1'])
        self.assertEqual((full.max_samples, full.max_steps, full.epochs), (0, -1, 2))
        self.assertTrue(smoke.smoke)
        self.assertEqual((smoke.batch_size, smoke.grad_accum), (2, 1))


if __name__ == '__main__':
    unittest.main()
