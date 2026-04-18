#!/usr/bin/env python3
"""
gpu_monitor.py — Parallel GPU-aware experiment scheduler for ExTra.

Monitors an 8-GPU shared node. Whenever 2 GPUs are free, launches the next
experiment from the queue. Multiple experiments can run in parallel if enough
GPUs are available. After each launch, waits MODEL_LOAD_WAIT_SEC seconds for
the model to load and claim GPU memory before checking availability again.

Usage:
    python gpu_monitor.py                # run in foreground
    python gpu_monitor.py --daemon       # detach and run in background
    python gpu_monitor.py --stop         # stop a running daemon
    python gpu_monitor.py --check-gpus  # print current GPU memory and exit
    python gpu_monitor.py --status      # show queue and running jobs

Edit EXPERIMENT_QUEUE below to configure which experiments to run.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — edit this section
# ─────────────────────────────────────────────────────────────────────────────

# Base directory for resolving relative script paths
SCRIPT_BASE_DIR = Path(__file__).parent / "verl" / "examples" / "experiments"

# Each entry: (script_path_relative_to_SCRIPT_BASE_DIR, n_gpus_required, name)
# Jobs run in order; multiple run in parallel whenever GPUs allow.
EXPERIMENT_QUEUE = [
    # ("ExTra_runs/01_grpo_baseline.sh",                   2, "grpo_baseline"),
    # ("ExTra_runs/02_extra_full.sh",                      2, "extra_full"),
    # ("ExTra_runs/03_grpo_entropy.sh",                    2, "grpo_entropy"),
    # ("ExTra_runs/04_ablation_no_curiosity_warmup50.sh",  2, "ablation_no_curiosity"),
    ("ExTra_runs/03_grpo_entropy.sh",                    2, "grpo_entropy"),
    # ("ExTra_runs/05_ablation_no_regeneration.sh",        2, "ablation_no_regen"),
    # ("02_extra_warmup_0.sh",                             2, "warmup_0"),
    # ("03_extra_warmup_20.sh",                            2, "warmup_20"),
    # ("04_extra_warmup_50.sh",                            2, "warmup_50"),
    # ("05_extra_warmup_200.sh",                           2, "warmup_200"),
]

TOTAL_GPUS = 8
POLL_INTERVAL_SEC = 120        # Seconds between GPU checks when idle
MODEL_LOAD_WAIT_SEC = 120     # Seconds to wait after launching before rechecking
                               # (lets the model load and claim GPU memory first)
LOG_FILE = Path(__file__).parent / "gpu_monitor.log"
PID_FILE = Path(__file__).parent / "gpu_monitor.pid"
LOG_DIR = Path.home() / "ExTra_logs"

# ─────────────────────────────────────────────────────────────────────────────


def setup_logging(to_file: bool = False) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(LOG_FILE)))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_gpus_with_processes() -> dict[int, list[int]]:
    """Return a mapping of GPU index -> list of PIDs with compute processes on that GPU."""
    # Build UUID -> index mapping
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    uuid_to_idx: dict[str, int] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(", ")
        uuid_to_idx[parts[1].strip()] = int(parts[0].strip())

    gpu_procs: dict[int, list[int]] = {i: [] for i in range(TOTAL_GPUS)}

    # Query compute processes per GPU
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(", ")
        idx = uuid_to_idx.get(parts[0].strip())
        if idx is not None:
            gpu_procs[idx].append(int(parts[1].strip()))
    return gpu_procs


def physically_free_gpus() -> list[int]:
    """GPUs with no compute processes running (process-based detection)."""
    gpu_procs = get_gpus_with_processes()
    return [i for i, pids in sorted(gpu_procs.items()) if not pids]


def pick_gpu_block(available: list[int], n: int) -> list[int] | None:
    """
    Pick n GPUs from the available list.
    Prefers GPUs within the same 4-GPU block (0-3, then 4-7) to minimise
    fragmentation. Falls back to any n available GPUs.
    """
    for block in [list(range(0, 4)), list(range(4, 8))]:
        candidates = [g for g in block if g in available]
        if len(candidates) >= n:
            return candidates[:n]
    if len(available) >= n:
        return list(available)[:n]
    return None


def launch_job(script_path: Path, gpu_list: list[int], exp_name: str) -> subprocess.Popen:
    """Launch a bash script with the given GPU assignment. Returns the Popen handle."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{exp_name}.log"
    gpu_str = ",".join(str(g) for g in gpu_list)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str  # overrides any hardcoded value in the script
    # Ensure the conda 'verl' env is used (sentence-transformers, ray, etc.)
    conda_bin = str(Path.home() / "miniconda3" / "envs" / "verl" / "bin")
    env["PATH"] = conda_bin + ":" + env.get("PATH", "")
    # Ensure `python3 -m verl.trainer.main_ppo` can find the verl package
    verl_root = str(Path(__file__).parent / "verl")
    env["PYTHONPATH"] = verl_root + ":" + env.get("PYTHONPATH", "")

    logging.info(f"LAUNCHING  {exp_name}  GPUs={gpu_str}  log={log_path}")

    log_file = open(str(log_path), "w")
    proc = subprocess.Popen(
        ["bash", str(script_path)],
        env=env,
        cwd=str(script_path.parent),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    # Attach metadata to the process object for bookkeeping
    proc._log_file = log_file
    proc._start_time = time.time()
    proc._exp_name = exp_name
    proc._gpu_list = gpu_list
    return proc


def fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def reap_finished(running_jobs: list, reserved_gpus: set) -> list:
    """
    Poll all running jobs. Remove finished ones, free their GPUs, return
    the updated list of still-running jobs.
    """
    still_running = []
    for proc in running_jobs:
        ret = proc.poll()
        if ret is not None:
            elapsed = time.time() - proc._start_time
            status = "SUCCEEDED" if ret == 0 else f"FAILED (exit={ret})"
            logging.info(f"FINISHED   {proc._exp_name}  {status}  elapsed={fmt_elapsed(elapsed)}")
            proc._log_file.close()
            for g in proc._gpu_list:
                reserved_gpus.discard(g)
        else:
            still_running.append(proc)
    return still_running


def run_monitor(queue: list[tuple[str, int, str]]) -> None:
    remaining = list(queue)
    running_jobs: list[subprocess.Popen] = []
    reserved_gpus: set[int] = set()

    logging.info(
        f"Starting GPU monitor  jobs={len(remaining)}"
        f"  poll={POLL_INTERVAL_SEC}s  model_load_wait={MODEL_LOAD_WAIT_SEC}s"
        f"  detection=process-based"
    )

    def handle_signal(signum, frame):
        logging.info(f"Signal {signum} received. Waiting for {len(running_jobs)} running job(s)...")
        for proc in running_jobs:
            proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while True:
        # ── 1. Reap finished jobs, free their GPUs ─────────────────────────
        running_jobs = reap_finished(running_jobs, reserved_gpus)

        # ── 2. Exit when all work is done ──────────────────────────────────
        if not running_jobs and not remaining:
            logging.info("All experiments complete. Exiting.")
            break

        # ── 3. Query GPU availability ──────────────────────────────────────
        try:
            phys_free = physically_free_gpus()
        except Exception as exc:
            logging.warning(f"nvidia-smi failed: {exc}. Retrying in {POLL_INTERVAL_SEC}s.")
            time.sleep(POLL_INTERVAL_SEC)
            continue

        # GPUs we can actually assign: physically free AND not reserved by our jobs
        available = [g for g in phys_free if g not in reserved_gpus]

        running_names = [p._exp_name for p in running_jobs]
        logging.info(
            f"GPU status  phys_free={phys_free}  reserved={sorted(reserved_gpus)}"
            f"  available={available}"
            f"  running={running_names}  queued={len(remaining)}"
        )

        # ── 4. Launch as many jobs as available GPUs allow ─────────────────
        launched_any = False
        while remaining and len(available) >= 2:
            script_rel, n_gpus, exp_name = remaining[0]
            script_path = SCRIPT_BASE_DIR / script_rel

            if not script_path.exists():
                logging.error(f"Script not found: {script_path}  — skipping '{exp_name}'")
                remaining.pop(0)
                continue

            gpu_block = pick_gpu_block(available, n_gpus)
            if gpu_block is None:
                # Not enough GPUs in any block; stop trying for this cycle
                logging.info(
                    f"Need {n_gpus} GPUs for '{exp_name}' but only {len(available)} available. Waiting."
                )
                break

            remaining.pop(0)
            try:
                proc = launch_job(script_path, gpu_block, exp_name)
                running_jobs.append(proc)
                reserved_gpus.update(gpu_block)
                # Remove just-reserved GPUs from available so next iteration
                # doesn't double-assign them
                available = [g for g in available if g not in reserved_gpus]
                launched_any = True
            except Exception as exc:
                logging.error(f"Failed to launch '{exp_name}': {exc}")

        # ── 5. Sleep: wait for model loading if we just launched, else poll ─
        if launched_any:
            logging.info(
                f"Waiting {MODEL_LOAD_WAIT_SEC}s for model(s) to load "
                f"before next GPU check..."
            )
            time.sleep(MODEL_LOAD_WAIT_SEC)
        else:
            time.sleep(POLL_INTERVAL_SEC)


def daemonize() -> None:
    """Fork twice to fully detach from the terminal."""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)

    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "r") as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open(str(LOG_FILE), "a") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

    PID_FILE.write_text(str(os.getpid()))
    logging.info(f"Daemon started  PID={os.getpid()}  log={LOG_FILE}")


def stop_daemon() -> None:
    if not PID_FILE.exists():
        print("No PID file found. Is the daemon running?")
        sys.exit(1)
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}.")
        PID_FILE.unlink(missing_ok=True)
    except ProcessLookupError:
        print(f"No process with PID {pid}. Removing stale PID file.")
        PID_FILE.unlink(missing_ok=True)


def check_gpus() -> None:
    try:
        gpu_procs = get_gpus_with_processes()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    free = physically_free_gpus()
    print(f"{'GPU':>4}  {'Processes':>10}  {'Status':>8}")
    print("-" * 34)
    for i in range(TOTAL_GPUS):
        pids = gpu_procs.get(i, [])
        status = "FREE" if i in free else "BUSY"
        pid_str = ",".join(str(p) for p in pids) if pids else "-"
        print(f"{i:>4}  {pid_str:>10}  {status:>8}")
    print(f"\n{len(free)}/{TOTAL_GPUS} GPUs free  (process-based detection)")


def show_status() -> None:
    print("=== Experiment Queue ===")
    for i, (script, n, name) in enumerate(EXPERIMENT_QUEUE):
        print(f"  [{i+1:2d}] {name:45s} ({n} GPUs)  {script}")
    if PID_FILE.exists():
        pid = PID_FILE.read_text().strip()
        print(f"\nDaemon running  PID={pid}  log={LOG_FILE}")
    else:
        print("\nNo daemon running.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel GPU-aware experiment scheduler for ExTra.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--daemon",     action="store_true", help="Run as background daemon")
    parser.add_argument("--stop",       action="store_true", help="Stop a running daemon")
    parser.add_argument("--check-gpus", action="store_true", help="Print GPU status and exit")
    parser.add_argument("--status",     action="store_true", help="Show queue and daemon status")
    args = parser.parse_args()

    if args.stop:
        stop_daemon()
        return
    if args.check_gpus:
        check_gpus()
        return
    if args.status:
        show_status()
        return

    if args.daemon:
        setup_logging(to_file=False)
        daemonize()
        setup_logging(to_file=True)
    else:
        setup_logging(to_file=False)

    run_monitor(EXPERIMENT_QUEUE)


if __name__ == "__main__":
    main()
