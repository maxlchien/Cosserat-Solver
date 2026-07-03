"""
Compare reference and computed seismograms for benchmark analysis.

This tool matches reference traces against computed traces using the benchmark
trace list and generates analytics reports with absolute/relative differences,
maximum values, and correlation coefficients.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import pearsonr


def read_seismogram(filename):
    """Read seismogram file (time, displacement)."""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]


def read_stations(filename):
    """Read STATIONS file."""
    stations = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                station = parts[0]
                network = parts[1]
                y = float(parts[2])
                x = float(parts[3])
                z = float(parts[5])
                stations.append(
                    {
                        "station": station,
                        "network": network,
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )
    return stations


def read_reference_locations(filename):
    """Read reference seismogram locations from params.yaml."""
    with open(filename) as f:
        params = yaml.safe_load(f)
    return [tuple(location) for location in params["seismogram_locations"]]


def normalize_coord_triple(y, x, z, ndigits=6):
    """Normalize 3D coordinates for stable dictionary key matching."""
    return (
        round(float(y), ndigits),
        round(float(x), ndigits),
        round(float(z), ndigits),
    )


def resolve_path(path_value, default_path):
    """Resolve a CLI path relative to the benchmark directory."""
    if path_value is None:
        return default_path
    path = Path(path_value)
    return path if path.is_absolute() else (Path(__file__).resolve().parent / path)


def load_trace_pairs_from_csv(trace_list_path):
    """Load reference/generated trace pairs from a CSV file."""
    trace_pairs = []
    with open(trace_list_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                msg = f"Invalid trace pair line: {line}"
                raise ValueError(msg)
            trace_pairs.append((parts[0], parts[1]))
    return trace_pairs


def load_reference_traces(results_dir, channel_extension="semd"):
    """
    Load reference seismograms from the benchmark output directory.

    Returns dict: {(station_name, channel): (time, displacement)}
    """
    traces = {}
    pattern = os.path.join(results_dir, f"*.{channel_extension}")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        parts = filename.split(".")
        if len(parts) >= 4:
            station = parts[1]
            channel = parts[3]
            time, displacement = read_seismogram(filepath)
            traces[(station, channel)] = (time, displacement)
    return traces


def load_computed_traces(results_dir, channel_extension="semd"):
    """
    Load computed seismograms from the benchmark output directory.

    Returns dict: {computed_station_id: {channel: (time, displacement)}}
    Example: {"S0001": {"XX": (time, disp), "XY": (time, disp), ...}}
    """
    traces = {}

    pattern = os.path.join(results_dir, f"*.{channel_extension}")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        parts = filename.split(".")
        if len(parts) >= 4:
            computed_id = parts[1]
            channel = parts[3]

            if computed_id not in traces:
                traces[computed_id] = {}

            time, displacement = read_seismogram(filepath)
            channel_key = channel[1:] if len(channel) > 1 else channel
            traces[computed_id][channel_key] = (time, displacement)

    return traces


def validate_and_match_traces(reference_traces, computed_traces, tolerance=1e-6):
    """
    Validate that time arrays match exactly between reference and computed traces.
    Match reference traces to computed traces by station index and channel letter.

    Returns: list of tuples (reference_key, computed_key, time_array) for matching pairs
    """
    reference_stations = sorted({station for (station, _) in reference_traces})
    computed_stations = sorted(computed_traces.keys())

    if len(reference_stations) != len(computed_stations):
        msg = (
            f"Number of stations mismatch: reference has {len(reference_stations)}, "
            f"computed has {len(computed_stations)}"
        )
        raise ValueError(msg)

    matched_pairs = []

    for reference_station, computed_station in zip(
        reference_stations, computed_stations, strict=False
    ):
        reference_channels = sorted(
            {channel for (s, channel) in reference_traces if s == reference_station}
        )
        computed_channels = sorted(computed_traces[computed_station].keys())

        if len(reference_channels) != len(computed_channels):
            msg = (
                f"Channel count mismatch for station pair {reference_station}/{computed_station}: "
                f"reference has {len(reference_channels)}, computed has {len(computed_channels)}"
            )
            raise ValueError(msg)

        for reference_channel, computed_channel in zip(
            reference_channels, computed_channels, strict=False
        ):
            time_reference, _ = reference_traces[(reference_station, reference_channel)]
            time_computed, _ = computed_traces[computed_station][computed_channel]

            if len(time_reference) != len(time_computed):
                msg = (
                    f"Time array length mismatch for {reference_station}/{reference_channel}: "
                    f"reference has {len(time_reference)}, computed has {len(time_computed)}"
                )
                raise ValueError(msg)

            dt_reference = np.mean(np.diff(time_reference))
            dt_computed = np.mean(np.diff(time_computed))

            if abs(dt_reference - dt_computed) > tolerance:
                msg = (
                    f"Timestep mismatch for {reference_station}/{reference_channel}: "
                    f"reference dt={dt_reference:.6f}, computed dt={dt_computed:.6f}"
                )
                raise ValueError(msg)

            if abs(time_reference[0] - time_computed[0]) > tolerance:
                msg = (
                    f"Start time mismatch for {reference_station}/{reference_channel}: "
                    f"reference t0={time_reference[0]:.6f}, computed t0={time_computed[0]:.6f}"
                )
                raise ValueError(msg)

            if abs(time_reference[-1] - time_computed[-1]) > tolerance:
                msg = (
                    f"End time mismatch for {reference_station}/{reference_channel}: "
                    f"reference t_end={time_reference[-1]:.6f}, computed t_end={time_computed[-1]:.6f}"
                )
                raise ValueError(msg)

            matched_pairs.append(
                (
                    (reference_station, reference_channel),
                    (computed_station, computed_channel),
                    time_reference,
                )
            )

    return matched_pairs


def compute_metrics(disp_ref, disp_comp):
    """
    Compute comparison metrics between two displacement arrays (same time grid).

    Returns dict with keys:
      - l1_normalized: l1 norm of difference normalized by l1 norm of computed
      - max_reference, max_computed: maximum absolute amplitudes
      - max_abs_diff: maximum absolute difference
      - rms_abs_diff: RMS absolute difference
      - max_rel_diff_pointwise: maximum point-by-point relative difference (%)
      - mean_rel_diff_pointwise: mean point-by-point relative difference (%)
      - rel_diff_overall: absolute difference divided by trace magnitude (%)
      - correlation: Pearson correlation coefficient
    """

    l1_error = np.sum(np.abs(disp_ref - disp_comp))
    l1_comp = np.sum(np.abs(disp_comp))
    l1_normalized = l1_error / l1_comp if l1_comp > 0 else float("inf")

    # Max absolute values
    max_ref = np.max(np.abs(disp_ref))
    max_comp = np.max(np.abs(disp_comp))

    # Absolute difference
    abs_diff = np.abs(disp_ref - disp_comp)
    max_abs_diff = np.max(abs_diff)
    rms_abs_diff = np.sqrt(np.mean(abs_diff**2))

    # Point-by-point relative difference (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff_pointwise = (
            np.abs((disp_ref - disp_comp) / (np.abs(disp_ref) + 1e-15)) * 100
        )
        rel_diff_pointwise = rel_diff_pointwise[np.isfinite(rel_diff_pointwise)]

    max_rel_diff_pointwise = (
        np.max(rel_diff_pointwise) if len(rel_diff_pointwise) > 0 else 0.0
    )
    mean_rel_diff_pointwise = (
        np.mean(rel_diff_pointwise) if len(rel_diff_pointwise) > 0 else 0.0
    )

    # Overall relative difference: mean of absolute differences / trace magnitude
    trace_magnitude = max_ref
    if trace_magnitude > 0:
        rel_diff_overall = (np.mean(abs_diff) / trace_magnitude) * 100
    else:
        rel_diff_overall = 0.0

    # Correlation
    try:
        corr, _ = pearsonr(disp_ref, disp_comp)
    except Exception:
        corr = np.nan

    # Normalized max absolute difference
    max_amplitude = max(max_ref, max_comp)
    norm_max_abs_diff = (
        (max_abs_diff / max_amplitude) * 100 if max_amplitude > 0 else 0.0
    )

    return {
        "l1_normalized": l1_normalized,
        "max_reference": max_ref,
        "max_computed": max_comp,
        "norm_max_abs_diff": norm_max_abs_diff,
        "max_abs_diff": max_abs_diff,
        "rms_abs_diff": rms_abs_diff,
        "max_rel_diff_pointwise": max_rel_diff_pointwise,
        "mean_rel_diff_pointwise": mean_rel_diff_pointwise,
        "rel_diff_overall": rel_diff_overall,
        "correlation": corr,
    }


def parse_reference_trace_label(reference_name):
    """Extract a compact station/channel label from a reference trace filename."""
    parts = Path(reference_name).name.split(".")
    if len(parts) >= 4:
        return parts[1], parts[3]
    return reference_name, ""


def generate_summary_report(all_metrics_by_channel, output_file):
    """Generate summary report of all comparisons across all channels."""
    with open(output_file, "w") as f:
        f.write("=" * 155 + "\n")
        f.write("TRACE COMPARISON: Reference vs Computed\n")
        f.write("=" * 155 + "\n\n")

        for channel_name in sorted(all_metrics_by_channel.keys()):
            all_metrics = all_metrics_by_channel[channel_name]

            f.write(f"\n{channel_name.upper()} CHANNEL\n")
            f.write("-" * 155 + "\n")
            f.write(
                f"{'Station':<10} {'Channel':<8} {'Norm l1':<18} {'NormMaxDiff(%)':<18} {'Max AbsDiff':<15} "
                f"{'RMS AbsDiff':<15} {'Max RelDiff(pp)(%)':<18} {'Mean RelDiff(pp)(%)':<18} "
                f"{'RelDiff(overall)(%)':<18} {'Correlation':<12} {'Max REF':<15} {'Max COMP':<15}\n"
            )
            f.write("-" * 155 + "\n")

            for (reference_name, _generated_name), metrics in sorted(
                all_metrics.items()
            ):
                station, channel = parse_reference_trace_label(reference_name)
                f.write(
                    f"{station:<10} {channel:<8} {metrics['l1_normalized']:<18.6f} "
                    f"{metrics['norm_max_abs_diff']:<18.6f} {metrics['max_abs_diff']:<15.6e} "
                    f"{metrics['rms_abs_diff']:<15.6e} "
                    f"{metrics['max_rel_diff_pointwise']:<18.6f} {metrics['mean_rel_diff_pointwise']:<18.6f} "
                    f"{metrics['rel_diff_overall']:<18.6f} {metrics['correlation']:<12.6f} "
                    f"{metrics['max_reference']:<15.6e} {metrics['max_computed']:<15.6e}\n"
                )

        f.write("\n" + "=" * 155 + "\n")
        f.write("LEGEND (Quality Metrics - Lower is Better):\n")
        f.write(
            "  Norm l1:               L1 norm of difference normalized by L1 norm of computed trace (used in testing with tolerance 1e-3)\n"
        )
        f.write(
            "  NormMaxDiff(%):        max(|REF-COMP|) / max(max|REF|, max|COMP|) as percentage\n"
        )
        f.write("  Max AbsDiff:           Maximum absolute difference between traces\n")
        f.write("  RMS AbsDiff:           Root-mean-square of absolute differences\n")
        f.write(
            "  Max/Mean RelDiff(pp):  Maximum/mean point-by-point relative difference as percentage\n"
        )
        f.write(
            "  RelDiff(overall):      Mean of absolute differences divided by trace magnitude (%)\n"
        )
        f.write(
            "  Correlation:           Pearson correlation coefficient (1.0 = perfect)\n"
        )
        f.write("\nAMPLITUDE (Reference):\n")
        f.write("  Max REF/COMP:         Maximum absolute amplitude value\n")
        f.write("=" * 155 + "\n")
        f.write(
            "  RelDiff(overall):      Sum of absolute differences divided by trace magnitude (%)\n"
        )
        f.write(
            "  Correlation:           Pearson correlation coefficient (1.0 = perfect)\n"
        )
        f.write("=" * 130 + "\n")


def generate_plots(
    matched_pairs, reference_traces, computed_traces, output_dir, limit=None
):
    """
    Generate comparison plots for matched trace pairs.

    Args:
        matched_pairs: list of (reference_key, computed_key, time) tuples from validate_and_match_traces
        reference_traces: dict of reference traces
        computed_traces: dict of computed traces
        output_dir: directory to save plots
        limit: maximum number of stations to plot (None = all)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if limit:
        matched_pairs = matched_pairs[:limit]

    # Group by reference station
    by_station = {}
    for ref_key, comp_key, time in matched_pairs:
        station = ref_key[0]
        if station not in by_station:
            by_station[station] = []
        by_station[station].append((ref_key, comp_key, time))

    # Create one figure per station
    for station in sorted(by_station.keys()):
        pairs = by_station[station]
        n_channels = len(pairs)

        fig = plt.figure(figsize=(15, 4 * n_channels))
        gs = gridspec.GridSpec(n_channels, 1, hspace=0.4)

        for idx, (reference_key, computed_key, time) in enumerate(pairs):
            _, disp_reference = reference_traces[reference_key]
            _, disp_computed = computed_traces[computed_key[0]][computed_key[1]]

            ax = fig.add_subplot(gs[idx])
            ax.plot(time, disp_reference, label="Reference", linewidth=1.5, alpha=0.8)
            ax.plot(time, disp_computed, label="Computed", linewidth=1.5, alpha=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Displacement")
            ax.set_title(f"Station {station} - Channel {reference_key[1]}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.savefig(
            os.path.join(output_dir, f"comparison_{station}.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare reference and generated seismogram traces using trace_list.csv."
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Path to the reference trace directory (default: ./reference_traces)",
    )
    parser.add_argument(
        "--generated-dir",
        type=str,
        default=None,
        help="Path to the generated trace directory (default: ./OUTPUT_FILES)",
    )
    parser.add_argument(
        "--trace-list",
        type=str,
        default=None,
        help="Path to the trace list CSV (default: ./trace_list.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: ./OUTPUT_FILES/analysis)",
    )
    parser.add_argument(
        "--channel",
        choices=["displacement", "rotation", "both"],
        default=None,
        help="Channel type to analyze: 'displacement' (.semd), 'rotation' (.semr), or 'both'. "
        "If not specified, analyzes both channels.",
    )
    parser.add_argument(
        "--plots", action="store_true", help="Generate comparison plots"
    )
    parser.add_argument(
        "--plot-limit",
        type=int,
        default=None,
        help="Maximum number of stations to plot (None = all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    reference_dir = resolve_path(args.reference_dir, base_dir / "reference_traces")
    generated_dir = resolve_path(args.generated_dir, base_dir / "OUTPUT_FILES")
    trace_list_path = resolve_path(args.trace_list, base_dir / "trace_list.csv")
    output_dir = resolve_path(args.output_dir, base_dir / "OUTPUT_FILES" / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    channels_to_process = (
        ["displacement", "rotation"]
        if args.channel in (None, "both")
        else [args.channel]
    )
    all_metrics_by_channel = {}
    trace_pairs = load_trace_pairs_from_csv(trace_list_path)

    if args.verbose:
        print(f"Loaded {len(trace_pairs)} trace pairs from {trace_list_path}")

    for channel in channels_to_process:
        channel_extension = "semd" if channel == "displacement" else "semr"
        channel_pairs = [
            pair
            for pair in trace_pairs
            if pair[0].endswith(channel_extension)
            and pair[1].endswith(channel_extension)
        ]

        if args.verbose:
            print(f"\n{'=' * 60}")
            print(f"Processing {channel} channel (*.{channel_extension})")
            print(f"{'=' * 60}")
            print(f"Using reference traces from: {reference_dir}")
            print(f"Using generated traces from: {generated_dir}")

        if not channel_pairs:
            if args.verbose:
                print(f"No {channel} trace pairs found in {trace_list_path}")
            continue

        all_metrics = {}
        skipped_pairs = []
        for reference_name, generated_name in channel_pairs:
            reference_path = reference_dir / reference_name
            generated_path = generated_dir / generated_name
            if not reference_path.exists():
                skipped_pairs.append(
                    (
                        reference_name,
                        generated_name,
                        f"missing reference: {reference_path}",
                    )
                )
                continue
            if not generated_path.exists():
                skipped_pairs.append(
                    (
                        reference_name,
                        generated_name,
                        f"missing generated: {generated_path}",
                    )
                )
                continue

            time_ref, disp_ref = read_seismogram(reference_path)
            time_gen, disp_gen = read_seismogram(generated_path)

            if len(time_ref) != len(time_gen):
                skipped_pairs.append(
                    (
                        reference_name,
                        generated_name,
                        f"trace length mismatch: {len(time_ref)} vs {len(time_gen)}",
                    )
                )
                continue

            if abs(np.mean(np.diff(time_ref)) - np.mean(np.diff(time_gen))) > 1e-6:
                skipped_pairs.append(
                    (reference_name, generated_name, "timestep mismatch")
                )
                continue

            all_metrics[(reference_name, generated_name)] = compute_metrics(
                disp_ref, disp_gen
            )

        if args.verbose and skipped_pairs:
            print(f"Skipped {len(skipped_pairs)} trace pair(s):")
            for reference_name, generated_name, reason in skipped_pairs:
                print(f"  - {reference_name} -> {generated_name}: {reason}")

        all_metrics_by_channel[channel] = all_metrics

        if args.plots and args.verbose:
            print("Plot generation is not enabled for trace-list based comparisons yet")

    summary_file = output_dir / "comparison_summary.txt"
    if args.verbose:
        print(f"\nWriting summary report to: {summary_file}")
    generate_summary_report(all_metrics_by_channel, summary_file)

    print(f"Summary report written to: {summary_file}")


if __name__ == "__main__":
    main()
