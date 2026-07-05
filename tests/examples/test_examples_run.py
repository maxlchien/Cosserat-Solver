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
def test_simulation_agreement(example: str) -> None:
    subprocess.run(
        ["snakemake", "clean", "-c1"],
        check=False,
        cwd=EXAMPLE_DIR / f"{example}",
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["snakemake", "-c1"],
        check=False,
        cwd=EXAMPLE_DIR / f"{example}",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Example {example} failed with error: {result.stderr}"
    )
