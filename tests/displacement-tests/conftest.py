from __future__ import annotations

import pytest

# read the data folder and get all the subfolders which have complete reference and param data

SIMULATIONS = ["benchmark"]


@pytest.fixture(params=SIMULATIONS, scope="session")
def simulation(request):
    return request.param


def pytest_addoption(parser):
    parser.addoption(
        "--displacement-tests",
        action="store_true",
        help="Run displacement tests. These are slow and run best using xdist.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--displacement-tests"):
        skip = pytest.mark.skip(reason="Pass --displacement-tests to run")
        for item in items:
            if "displacement_test" in item.keywords:
                item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "displacement_test: mark test as a displacement test, which is slow and should be run with --displacement-tests",
    )
