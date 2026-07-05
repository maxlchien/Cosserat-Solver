from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_seismogram(filename):
    """Read a seismogram file as (time, displacement)."""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]


def resolve_path(path_value, default_path):
    """Resolve a CLI path relative to the benchmark directory."""
    if path_value is None:
        return default_path
    path = Path(path_value)
    return path if path.is_absolute() else (Path(__file__).resolve().parent / path)


def load_trace_pairs_from_csv(trace_list_path):
    """Load reference/generated trace pairs from the benchmark trace list."""
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot benchmark seismogram comparisons."
    )
    parser.add_argument(
        "--channel",
        choices=["displacement", "rotation", "both"],
        default=None,
        help="Choose which trace type to load: displacement (.semd), rotation (.semr), or both. If omitted, both are processed.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure-level title shown above all subplots.",
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
        "--params",
        type=str,
        default=None,
        help="Path to the benchmark params YAML (default: ./params.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: ./OUTPUT_FILES/<channel>_plot.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    reference_dir = resolve_path(args.reference_dir, base_dir / "reference_traces")
    generated_dir = resolve_path(args.generated_dir, base_dir / "OUTPUT_FILES")
    trace_list_path = resolve_path(args.trace_list, base_dir / "trace_list.csv")

    channels_to_process = (
        ["displacement", "rotation"]
        if args.channel in (None, "both")
        else [args.channel]
    )
    trace_pairs = load_trace_pairs_from_csv(trace_list_path)

    for channel in channels_to_process:
        channel_extension = "semd" if channel == "displacement" else "semr"
        filtered_pairs = [
            (reference_name, generated_name)
            for reference_name, generated_name in trace_pairs
            if reference_name.endswith(channel_extension)
            and generated_name.endswith(channel_extension)
        ]

        if not filtered_pairs:
            print(f"No {channel} trace pairs found in {trace_list_path}")

        reference_traces = {}
        generated_traces = {}
        skipped_pairs = []
        for reference_name, generated_name in filtered_pairs:
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

            reference_traces[reference_name] = (time_ref, disp_ref)
            generated_traces[generated_name] = (time_gen, disp_gen)

        if not reference_traces:
            print(
                f"No comparable {channel} traces were loaded from the benchmark output."
            )
            continue

        if skipped_pairs:
            for reference_name, generated_name, reason in skipped_pairs:
                print(f"Skipping {reference_name} -> {generated_name}: {reason}")

        fig, axes = plt.subplots(
            len(filtered_pairs),
            1,
            figsize=(12, 2.8 * len(filtered_pairs)),
            squeeze=False,
        )
        if args.title:
            fig.suptitle(args.title, fontsize=12)
        else:
            fig.suptitle(
                f"{channel.title()} traces from benchmark trace list", fontsize=12
            )

        for ax, (reference_name, generated_name) in zip(
            axes.ravel(), filtered_pairs, strict=False
        ):
            if (
                reference_name not in reference_traces
                or generated_name not in generated_traces
            ):
                continue
            time_ref, disp_ref = reference_traces[reference_name]
            time_gen, disp_gen = generated_traces[generated_name]

            scale = max(np.max(np.abs(disp_ref)), np.max(np.abs(disp_gen)), 1.0)
            ax.plot(time_ref, disp_ref / scale, color="tab:blue", label="reference")
            ax.plot(
                time_gen,
                disp_gen / scale,
                color="tab:orange",
                linestyle="--",
                label="generated",
            )
            ax.set_title(f"{reference_name} -> {generated_name}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized displacement")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if args.output:
            output_path = Path(args.output)
            if output_path.suffix:
                output_path = output_path.with_name(
                    f"{output_path.stem}_{channel}{output_path.suffix}"
                )
            else:
                output_path = output_path / f"{channel}_plot.png"
        else:
            output_path = base_dir / "OUTPUT_FILES" / f"{channel}_plot.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {channel} plot to {output_path}")


if __name__ == "__main__":
    main()
