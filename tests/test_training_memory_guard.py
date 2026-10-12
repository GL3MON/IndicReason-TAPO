"""Exercise the RAM watchdog without importing ML libraries or allocating RAM."""
import contextlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from indic_reasoning.runtime import memory_guard as guard


class MemoryGuardTests(unittest.TestCase):
    def test_detects_ancestor_limit_even_when_leaf_is_unlimited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / "jobs" / "training"
            leaf.mkdir(parents=True)
            (root / "memory.max").write_text(str(16 * guard.GIB))
            (root / "memory.current").write_text("0")
            (leaf / "memory.max").write_text("max")
            (leaf / "memory.current").write_text("0")
            membership = root / "membership"
            membership.write_text("0::/jobs/training\n")
            self.assertEqual(guard.discover_limits(root, membership),
                             [(root / "memory.current", 16 * guard.GIB)])
            self.assertEqual(guard.threshold(16 * guard.GIB), 16 * guard.GIB)
            self.assertEqual(guard.threshold(32 * guard.GIB), 16 * guard.GIB)
            self.assertEqual(guard.threshold(6 * guard.GIB), 6 * guard.GIB)

    def test_v1_and_namespace_parent_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mount = root / "memory"
            mount.mkdir()
            (mount / "memory.limit_in_bytes").write_text(str(8 * guard.GIB))
            (mount / "memory.usage_in_bytes").write_text("0")
            membership = root / "membership"
            membership.write_text("5:memory:/../../outside\n")
            self.assertEqual(guard.discover_limits(root, membership),
                             [(mount / "memory.usage_in_bytes", 8 * guard.GIB)])

    def test_refuses_pressure_missing_limits_and_monitor_errors_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current"
            current.write_text(str(16 * guard.GIB))
            with patch.object(guard.subprocess, "Popen") as spawn, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(guard.run_guarded(["unused"], [(current, 16 * guard.GIB)]), 75)
                self.assertEqual(guard.run_guarded(["unused"], []), 75)
                current.unlink()
                self.assertEqual(guard.run_guarded(["unused"], [(current, 16 * guard.GIB)]), 75)
                spawn.assert_not_called()

    def test_disabled_cutoff_does_not_read_limits_but_keeps_supervisor(self):
        with patch.object(guard, "discover_limits", side_effect=AssertionError("must not read cgroups")), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(guard.run_guarded([sys.executable, "-c", "raise SystemExit(7)"],
                                              poll_interval=.01, monitor_memory=False), 7)

    def test_previous_cutoff_is_now_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current"
            current.write_text(str(13 * guard.GIB))
            self.assertIsNone(guard.pressure([(current, 16 * guard.GIB)]))
            current.write_text(str(16 * guard.GIB))
            self.assertIn("16.00 GiB training cutoff", guard.pressure([(current, 16 * guard.GIB)]))

    def test_cpu_and_compile_caps_preserve_device_selection(self):
        env = guard.worker_environment({"OMP_NUM_THREADS": "64", "MKL_NUM_THREADS": "1",
                                        "MAX_JOBS": "32", "CUDA_VISIBLE_DEVICES": "MIG-test"})
        self.assertEqual(env["OMP_NUM_THREADS"], "2")
        self.assertEqual(env["MKL_NUM_THREADS"], "1")
        self.assertEqual(env["MAX_JOBS"], "1")
        self.assertEqual(env["TORCHINDUCTOR_COMPILE_THREADS"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "MIG-test")

    def test_normal_exit_code_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current"
            current.write_text("0")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(guard.run_guarded([sys.executable, "-c", "raise SystemExit(7)"],
                                                  [(current, 16 * guard.GIB)], .01), 7)

    def test_training_main_starts_guard_before_dataset_import(self):
        import train_tapo_trl
        with patch.object(train_tapo_trl, "parse_args") as parse, \
             patch.object(guard, "supervise_training", return_value=75):
            parse.return_value.self_test = False
            self.assertEqual(train_tapo_trl.main(), 75)

    def integration(self, stop, separate_group=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current"
            current.write_text("0")
            pid_file = root / "worker.pid"
            # The worker is a tiny sleeping Python process, not an ML workload.
            job = ("import os,subprocess,sys,time;from pathlib import Path;"
                   "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
                   "Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)")
            if separate_group:
                job = job.replace("'import time;time.sleep(30)'])", "'import time;time.sleep(30)'],preexec_fn=os.setpgrp)")
            supervisor = ("import sys;from pathlib import Path;"
                          "from indic_reasoning.runtime.memory_guard import run_guarded,GIB;"
                          "raise SystemExit(run_guarded([sys.executable,'-c',sys.argv[3],sys.argv[2]],"
                          "[(Path(sys.argv[1]),16*GIB)],.01))")
            process = subprocess.Popen([sys.executable, "-c", supervisor, str(current), str(pid_file), job],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            worker = None
            try:
                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(process.communicate()[1])
                    time.sleep(.01)
                self.assertTrue(pid_file.exists(), "dummy worker failed to start")
                worker = int(pid_file.read_text())
                if stop == "pressure":
                    current.write_text(str(16 * guard.GIB))
                elif stop == "monitor_failure":
                    current.unlink()
                else:
                    process.send_signal(signal.SIGTERM)
                _, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 143 if stop == "signal" else 75, stderr)
                if stop == "pressure":
                    self.assertIn("stopping training and its workers", stderr)
                # A zombie has exited; it can remain until the container init reaps it.
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    stat = Path(f"/proc/{worker}/stat")
                    if not stat.exists() or stat.read_text().split(")", 1)[1].split()[0] == "Z":
                        break
                    time.sleep(.01)
                else:
                    self.fail("training worker survived watchdog shutdown")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=5)
                if worker:
                    try:
                        os.kill(worker, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_pressure_stops_worker_group(self):
        self.integration("pressure")

    def test_pressure_stops_worker_in_separate_process_group(self):
        self.integration("pressure", separate_group=True)

    def test_monitor_failure_stops_worker_group(self):
        self.integration("monitor_failure")

    def test_sigterm_stops_worker_group(self):
        self.integration("signal")


if __name__ == "__main__":
    unittest.main()
