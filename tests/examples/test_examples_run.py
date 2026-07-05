from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "examples"

EXAMPLES = ["cosserat_3d_benchmark"]


@pytest.fixture(params=EXAMPLES)
def example(request):
    """Fixture to provide example names for parameterized tests."""
    return request.param


# does not check for correctness, only that the example runs
@pytest.mark.displacement_test
@pytest.mark.timeout(60)  # Set a timeout of 1 minute for each example
def test_simulation_agreement(example: str, max_cores_per_worker: int) -> None:
    subprocess.run(
        ["snakemake", "clean", "-c1"],
        check=False,
        cwd=EXAMPLE_DIR / f"{example}",
        capture_output=True,
        text=True,
    )
    cores_to_request = min(4, max_cores_per_worker)  # Limit to 4 cores for the test
    result = subprocess.run(
        ["snakemake", f"-c{cores_to_request}"],
        check=False,
        cwd=EXAMPLE_DIR / f"{example}",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Example {example} failed with error: {result.stderr}"
    )
