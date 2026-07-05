from __future__ import annotations

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--displacement-tests",
        action="store_true",
        help="Run displacement tests. These are slow and run best using xdist.",
    )
    parser.addoption(
        "--max-cores",
        action="store",
        help="Maximum number of cores to be used across all tests.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--displacement-tests"):
        skip = pytest.mark.skip(reason="Pass --displacement-tests to run")
        for item in items:
            if "displacement_test" in item.keywords:
                item.add_marker(skip)
    else:
        skip = pytest.mark.skip(
            reason="--displacement-tests was passed, so skipping non-displacement tests"
        )
        for item in items:
            if "displacement_test" not in item.keywords:
                item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "displacement_test: mark test as a displacement test, which is slow and should be run with --displacement-tests",
    )


@pytest.fixture(scope="session")
def max_cores_per_worker(request):
    """
    Fixture to get the maximum number of cores per worker from pytest command line options.
    Defaults to 1 if not specified.
    """
    budget = len(os.sched_getaffinity(0))  # Default to the number of available cores
    if request.config.getoption("--max-cores"):
        budget = min(int(request.config.getoption("--max-cores")), budget)

    n_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    max_cores = budget // n_workers
    if max_cores < 1:
        msg = (
            f"Not enough cores available for the number of workers. "
            f"Budget: {budget}, Workers: {n_workers}, Max cores per worker: {max_cores}"
        )
        raise ValueError(msg)
    return max_cores
