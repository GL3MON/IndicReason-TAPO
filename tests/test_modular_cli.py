import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from indic_reasoning.runtime import memory_guard
from indic_reasoning.evaluation.scoring import summarize_results

class ModularCLITests(unittest.TestCase):
    def test_training_command_self_test_needs_no_gpu(self):
        result=subprocess.run([sys.executable,'-m','indic_reasoning','train','--self-test'],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('Self-test passed',result.stdout)

    def test_supervisor_preserves_module_dispatch_and_python_flags(self):
        invocation=[sys.executable,'-u','-m','indic_reasoning','train','--no-memory-guard']
        with patch.object(sys,'orig_argv',invocation),patch.dict(os.environ,{memory_guard.CHILD_MARKER:'0'}),patch.object(memory_guard,'run_guarded',return_value=0) as run:
            self.assertEqual(memory_guard.supervise_training(False),0)
            run.assert_called_once_with(invocation,monitor_memory=False)

    def test_extraction_and_correctness_are_distinct_metrics(self):
        rows=[dict(completion=r'wrong \\boxed{3}',correct=False,answer_correct_without_format_requirement=False,translation_reward=0.5,valid_translation=True,finish_reason='stop'),dict(completion='no answer',correct=False,answer_correct_without_format_requirement=False,translation_reward=0.0,valid_translation=False,finish_reason='length')]
        summary=summarize_results(rows)
        self.assertEqual(summary['accuracy'],0)
        self.assertEqual(summary['extractable_answers'],1)
        self.assertEqual(summary['truncated'],1)

if __name__=='__main__':unittest.main()
