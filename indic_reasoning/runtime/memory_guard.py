"""Standard-library-only container RAM watchdog, started before ML imports."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3
CHILD_MARKER = "TAPO_MEMORY_GUARD_CHILD"


def discover_limits(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Find finite v2/v1 limits, including limits imposed by ancestor cgroups."""
    directories = {root, root / "memory"}
    if membership.exists():
        for line in membership.read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if controllers and "memory" not in controllers.split(","):
                continue
            mount = root / "memory" if controllers else root
            candidate = mount / relative.lstrip("/")
            # Namespace-relative paths can contain '..'; never escape the mount.
            if ".." in candidate.relative_to(mount).parts:
                continue
            while True:
                directories.add(candidate)
                if candidate == mount:
                    break
                candidate = candidate.parent
    limits = []
    for directory in sorted(directories):
        for maximum, current in (("memory.max", "memory.current"),
                                 ("memory.limit_in_bytes", "memory.usage_in_bytes")):
            maximum, current = directory / maximum, directory / current
            if not maximum.exists() or not current.exists():
                continue
            value = maximum.read_text().strip()
            if value == "max":
                continue
            value = int(value)
            if 0 < value < 2 ** 60:  # v1 uses a huge integer for unlimited.
                limits.append((current, value))
    return limits


def threshold(limit):
    # User-requested 16 GiB cutoff, bounded by each actual container limit.
    # At a 16 GiB hard limit this leaves no headroom for watchdog polling.
    return min(limit, 16 * GIB)


def worker_environment(environ):
    env = dict(environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        try:
            value = int(env.get(name, "2"))
        except ValueError:
            value = 2
        env[name] = str(max(1, min(value, 2)))
    for name in ("MAX_JOBS", "TORCHINDUCTOR_COMPILE_THREADS", "CMAKE_BUILD_PARALLEL_LEVEL"):
        env[name] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env[CHILD_MARKER] = "1"
    return env


def pressure(limits):
    for current, limit in limits:
        used = int(current.read_text().strip())
        cutoff = threshold(limit)
        if used >= cutoff:
            return (f"container RAM {used / GIB:.2f} GiB reached the "
                    f"{cutoff / GIB:.2f} GiB training cutoff "
                    f"(container limit {limit / GIB:.2f} GiB; {current})")
    return None


def kill_group(process):
    # Popen(start_new_session=True) makes the child's PID its session ID.
    # Ninja creates additional process groups; include all of those groups.
    # An unrelated editor/server process cannot share this new session.
    for _ in range(3):
        groups = {process.pid}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                pid = int(entry.name)
                if os.getsid(pid) == process.pid:
                    groups.add(os.getpgid(pid))
            except ProcessLookupError:
                continue
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Rescan for children forked while the first scan was in progress.
        time.sleep(0.01)
    process.wait()


def run_guarded(command, limits=None, poll_interval=0.2, monitor_memory=True):
    limits = (discover_limits() if limits is None else limits) if monitor_memory else []
    if not monitor_memory:
        print("Memory guard: RAM cutoff disabled; CPU limits and worker cleanup remain active.",
              file=sys.stderr, flush=True)
    if monitor_memory and not limits:
        print("Memory guard: no finite container RAM limit detected; refusing to "
              "start. Run training in a container with a memory limit.", file=sys.stderr, flush=True)
        return 75
    try:
        reason = pressure(limits)
    except (OSError, ValueError) as exc:
        print(f"Memory guard: cannot read RAM usage: {exc}; refusing to start.",
              file=sys.stderr, flush=True)
        return 75
    if reason:
        print(f"Memory guard: {reason}; refusing to start.", file=sys.stderr, flush=True)
        return 75
    for current, limit in limits:
        print(f"Memory guard: watching {current}; limit={limit / GIB:.2f} GiB, "
              f"stop at {threshold(limit) / GIB:.2f} GiB. CPU threads <=2; compile workers=1.",
              file=sys.stderr, flush=True)
    requested_signal = []
    previous = {}
    def request_stop(signum, frame):
        requested_signal.append(signum)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request_stop)
    process = None
    try:
        process = subprocess.Popen(command, env=worker_environment(os.environ),
                                   start_new_session=True)
        while True:
            if requested_signal:
                return 128 + requested_signal[0]
            code = process.poll()
            if code is not None:
                return code if code >= 0 else 128 - code
            try:
                reason = pressure(limits)
            except (OSError, ValueError) as exc:
                reason = f"RAM monitor failed: {exc}"
            if reason:
                print(f"Memory guard: {reason}; stopping training and its workers. "
                      "The current step will not be saved; resume from the last completed checkpoint.",
                      file=sys.stderr, flush=True)
                return 75
            time.sleep(poll_interval)
    finally:
        if process is not None:
            kill_group(process)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def supervise_training(monitor_memory=True):
    """Return None inside the worker, or its exit code in the supervisor."""
    if os.environ.get(CHILD_MARKER) == "1":
        return None
    return run_guarded([sys.executable, *sys.orig_argv[1:]],
                       monitor_memory=monitor_memory)
