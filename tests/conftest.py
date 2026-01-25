from __future__ import annotations

import itertools
import os
import time

import pytest
from mpmath import mp

from cosserat_solver import consts

mp.dps = consts.TEST_PRECISION

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


# --- Predefined material parameter sets ---
MATERIAL_PARAMETER_SETS = [
    {
        "rho": mp.mpf("1e3"),
        "lam": mp.mpf("1e5"),
        "mu": mp.mpf("1e5"),
        "nu": mp.mpf("1e4"),
        "J": mp.mpf("1"),
        "lam_c": mp.mpf("1e5"),
        "mu_c": mp.mpf("1e5"),
        "nu_c": mp.mpf("1e4"),
    },
    {
        "rho": mp.mpf("1e4"),
        "lam": mp.mpf("1e6"),
        "mu": mp.mpf("5e6"),
        "nu": mp.mpf("1e5"),
        "J": mp.mpf("1e2"),
        "lam_c": mp.mpf("2e6"),
        "mu_c": mp.mpf("3e6"),
        "nu_c": mp.mpf("5e5"),
    },
    {
        "rho": mp.mpf("1e5"),
        "lam": mp.mpf("1e7"),
        "mu": mp.mpf("1e7"),
        "nu": mp.mpf("1e6"),
        "J": mp.mpf("1e6"),
        "lam_c": mp.mpf("1e7"),
        "mu_c": mp.mpf("1e7"),
        "nu_c": mp.mpf("1e6"),
    },
    {
        "rho": mp.mpf("1e6"),
        "lam": mp.mpf("1e8"),
        "mu": mp.mpf("5e7"),
        "nu": mp.mpf("1e7"),
        "J": mp.mpf("1e3"),
        "lam_c": mp.mpf("1e8"),
        "mu_c": mp.mpf("1e8"),
        "nu_c": mp.mpf("1e7"),
    },
    {
        "rho": mp.mpf("1e8"),
        "lam": mp.mpf("5e5"),
        "mu": mp.mpf("2e8"),
        "nu": mp.mpf("3e6"),
        "J": mp.mpf("1e4"),
        "lam_c": mp.mpf("4e7"),
        "mu_c": mp.mpf("6e7"),
        "nu_c": mp.mpf("8e6"),
    },
]

# --- Define k and omega ranges ---
K_VALUES = [
    mp.mpf("1e-1"),
    mp.mpf("1.0"),
    mp.mpf("10.0"),
    mp.mpf("100.0"),
    mp.mpf("1000.0"),
]


@pytest.fixture(params=K_VALUES, scope="session")
def k_value(request):
    return request.param


OMEGA_VALUES = [mp.mpf("1e1"), mp.mpf("1e2"), mp.mpf("1e3"), mp.mpf("1e4")]


@pytest.fixture(params=OMEGA_VALUES, scope="session")
def omega_value(request):
    return request.param


# --- Fixture for material parameters ---
@pytest.fixture(params=MATERIAL_PARAMETER_SETS, scope="session")
def material_parameters(request):
    """Fixture providing predefined sets of material parameters."""
    return request.param


# --- Fixture for (k, omega) combinations ---
@pytest.fixture(
    params=[
        {"k": k, "omega": omega}
        for k, omega in itertools.product(K_VALUES, OMEGA_VALUES)
    ],
    scope="session",
)
def wave_parameters(request):
    """Fixture providing combinations of wavenumber and frequency."""
    return request.param


@pytest.fixture(params=[consts.PLUS_BRANCH, -consts.PLUS_BRANCH], scope="session")
def branch(request):
    """Fixture providing the branch of the dispersion relation."""
    return request.param
