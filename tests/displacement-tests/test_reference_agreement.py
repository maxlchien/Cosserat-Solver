from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr

TEST_DIR = Path(__file__).resolve().parent

TOL = 1e-3  # Default tolerance for normalized error


@pytest.mark.displacement_test
@pytest.mark.timeout(300)  # Set a timeout of 5 minutes for each test
def test_simulation_agreement(simulation: str) -> None:
    subprocess.run(
        ["snakemake", "clean", "-c1"],
        check=False,
        cwd=TEST_DIR / f"data/{simulation}",
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["snakemake", "-c1"],
        check=False,
        cwd=TEST_DIR / f"data/{simulation}",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Simulation {simulation} failed with error: {result.stderr}"
    )

    statistics, pass_fail = analyze_agreement(simulation, TOL)
    num_failed = statistics.get(
        "num_failed", ValueError("num_failed not found in statistics")
    )

    if not pass_fail:
        error_messages = []
        for trace_name, stats in statistics.items():
            if stats["normalized_error"] > TOL:
                error_messages.append(
                    f"Trace {trace_name} failed with normalized error {stats['normalized_error']:.2e}, "
                    f"L1 error: {stats['l1_error']:.2e}, L1 norm: {stats['l1_norm']:.2e}, "
                    f"Pearson correlation: {stats['pearson_correlation']:.4f}, "
                    f"Max error: {stats['max_error']:.2e}, Max norm: {stats['max_norm']:.2e}, "
                    f"Max error norm: {stats['max_error_norm']:.2e}"
                )
        pytest.fail(
            f"Simulation {simulation} failed agreement test:\n"
            f"\n Number of failed traces: {num_failed}\n" + "\n".join(error_messages)
        )


def analyze_agreement(
    simulation_name: str, tolerance: float = 1e-3
) -> tuple[dict, bool]:
    """
    Compare generated traces against a reference trace.
    If the accumulated error (normalized by the l1 norm of the trace) is above tolerance, the test fails.

    Returns:
        A tuple containing a dictionary of statistics and a boolean indicating pass/fail.
    """
    # load the filenames
    trace_list_path = TEST_DIR / f"data/{simulation_name}/trace_list.csv"
    if not os.path.exists(trace_list_path):
        msg = f"Trace list file not found: {trace_list_path}"
        raise FileNotFoundError(msg)

    def validate_trace_line(line: str) -> tuple[str, str]:
        parts = line.strip().split(",")
        if len(parts) != 2:
            msg = f"Invalid line in trace_list.csv: {line.strip()}"
            raise ValueError(msg)
        return parts[0], parts[1]

    with open(trace_list_path) as f:
        trace_filenames = [validate_trace_line(line) for line in f.readlines()]

    statistics = {}
    fail = False
    num_failed = 0
    for ref, computed in trace_filenames:
        # load the traces
        ref_path = TEST_DIR / f"data/{simulation_name}/reference_traces/{ref}"
        computed_path = TEST_DIR / f"data/{simulation_name}/OUTPUT_FILES/{computed}"
        if not os.path.exists(ref_path):
            msg = f"Reference trace file not found: {ref_path}"
            raise FileNotFoundError(msg)
        if not os.path.exists(computed_path):
            msg = f"Computed trace file not found: {computed_path}"
            raise FileNotFoundError(msg)
        ref_trace = np.loadtxt(ref_path)
        computed_trace = np.loadtxt(computed_path)

        # check that the traces are the same length
        if ref_trace.shape != computed_trace.shape:
            msg = f"Trace length mismatch: {ref_path} has shape {ref_trace.shape}, but {computed_path} has shape {computed_trace.shape}"
            raise ValueError(msg)

        # check that the times are close to each other
        ref_times = ref_trace[:, 0]
        computed_times = computed_trace[:, 0]
        if not np.allclose(ref_times, computed_times, atol=1e-6):
            msg = f"Time mismatch between {ref_path} and {computed_path}"
            raise ValueError(msg)

        # find the l1 norm of the error and l1 norm of the computed trace
        error = np.abs(ref_trace[:, 1] - computed_trace[:, 1])
        l1_error = np.sum(error)
        l1_norm = np.sum(np.abs(computed_trace[:, 1]))
        normalized_error = l1_error / l1_norm if l1_norm > 0 else float("inf")

        # run auxiliary statistics: pearson correlation, maximum error divided by max norm of traces
        pearson_corr, _ = pearsonr(ref_trace[:, 1], computed_trace[:, 1])
        max_error = np.max(error)
        max_norm = max(
            np.max(np.abs(ref_trace[:, 1])), np.max(np.abs(computed_trace[:, 1]))
        )
        max_error_norm = max_error / max_norm if max_norm > 0 else float("inf")

        statistics[os.path.basename(ref_path)] = {
            "l1_error": l1_error,
            "l1_norm": l1_norm,
            "normalized_error": normalized_error,
            "pearson_correlation": pearson_corr,
            "max_error": max_error,
            "max_norm": max_norm,
            "max_error_norm": max_error_norm,
        }
        if normalized_error > tolerance:
            fail = True
            num_failed += 1

    # return a dict and boolean for pass/fail
    statistics["num_failed"] = num_failed
    return statistics, not fail
