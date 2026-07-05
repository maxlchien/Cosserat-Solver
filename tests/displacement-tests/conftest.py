from __future__ import annotations

import pytest

# read the data folder and get all the subfolders which have complete reference and param data

SIMULATIONS = ["cosserat_3d_benchmark", "elastic_3d_benchmark"]


@pytest.fixture(params=SIMULATIONS, scope="session")
def simulation(request):
    return request.param
