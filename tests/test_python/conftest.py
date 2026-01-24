from __future__ import annotations

import os
import time

import pytest

# Store timing data in a shared location
_timing_file = ".pytest_timings.txt"


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Initialize timing at session start"""
    session.start_cpu_time = time.process_time()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Record CPU time at session end"""
    cpu_time = time.process_time() - session.start_cpu_time
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")

    # Append timing data
    with open(_timing_file, "a") as f:
        f.write(f"{worker_id},{cpu_time}\n")

    # If master process, read and sum all timings
    if worker_id == "master":
        time.sleep(0.5)  # Give workers time to write
        total = 0
        try:
            with open(_timing_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        total += float(parts[1])
            print(f"\n\nTotal CPU time across all workers: {total:.2f}s")
        finally:
            os.remove(_timing_file)
