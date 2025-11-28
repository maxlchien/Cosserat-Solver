# from networkx import omega
from __future__ import annotations

import glob
import os
import sys

import matplotlib as mpl
import numpy as np
import obspy

# Set matplotlib gui off
mpl.use("Agg")
import re

# TODO: replace utilities module
from pathlib import Path

import matplotlib.pyplot as plt

# read arguments
# get output directory
output_dir = sys.argv[1] if len(sys.argv) > 1 else "OUTPUT_FILES"


def get_traces(directory, ext=None):
    traces = []
    if ext is None:
        files = glob.glob(directory + "/*.sem*")
    else:
        files = glob.glob(directory + f"/*.{ext}")  # e.g. semd, semr, semir, semc
    ## iterate over all seismograms
    for filename in files:
        station_name = os.path.splitext(filename)[0]
        network, station, location, channel = station_name.split("/")[-1].split(".")
        trace = np.loadtxt(filename, delimiter=" ")
        starttime = trace[0, 0]
        dt = trace[1, 0] - trace[0, 0]
        traces.append(
            obspy.Trace(
                trace[:, 1],
                {
                    "network": network,
                    "station": station,
                    "location": location,
                    "channel": channel,
                    "starttime": starttime,
                    "delta": dt,
                },
            )
        )

    return obspy.Stream(traces)


def plot_station_with_arrivals(station_num, subtitle="", apply_filter=False):  # noqa: ARG001
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    # Create a list to store traces in the desired order
    ordered_traces = []
    trace_labels = [
        "Displacement X",
        "Displacement Z",
        "Rotation",
        "Intrinsic Rotation",
        "Curl",
    ]

    # Load displacement traces (X and Z components)
    displacement_stream = get_traces(f"{output_dir}", "semd")
    station_displacement = displacement_stream.select(station=f"S{station_num:04d}")

    # Add X component displacement
    x_trace = station_displacement.select(component="X")
    if len(x_trace) > 0:
        ordered_traces.append(x_trace[0])
    else:
        ordered_traces.append(None)

    # Add Z component displacement
    z_trace = station_displacement.select(component="Z")
    if len(z_trace) > 0:
        ordered_traces.append(z_trace[0])
    else:
        ordered_traces.append(None)

    # Load rotation traces (Y component)
    rotation_stream = get_traces(f"{output_dir}", "semr")
    station_rotation = rotation_stream.select(station=f"S{station_num:04d}")
    y_rotation = station_rotation.select(component="Y")
    if len(y_rotation) > 0:
        ordered_traces.append(y_rotation[0])
    else:
        ordered_traces.append(None)

    # Load intrinsic rotation traces (Y component)
    intrinsic_stream = get_traces(f"{output_dir}", "semir")
    station_intrinsic = intrinsic_stream.select(station=f"S{station_num:04d}")
    y_intrinsic = station_intrinsic.select(component="Y")
    if len(y_intrinsic) > 0:
        ordered_traces.append(y_intrinsic[0])
    else:
        ordered_traces.append(None)

    # Load curl traces (Y component)
    curl_stream = get_traces(f"{output_dir}", "semc")
    station_curl = curl_stream.select(station=f"S{station_num:04d}")
    y_curl = station_curl.select(component="Y")
    if len(y_curl) > 0:
        ordered_traces.append(y_curl[0])
    else:
        ordered_traces.append(None)

    # Check if we have at least one trace
    if all(trace is None for trace in ordered_traces):
        print(f"No traces found for station {station_num}")
        plt.close(fig)
        return

    # Plot each trace
    for i, (trace, label) in enumerate(zip(ordered_traces, trace_labels, strict=False)):
        ax = axes[i]

        if trace is not None:
            # Calculate origin time and time reference
            trace_start_time = trace.stats.starttime
            # Calculate origin time (trace start + half period offset)
            origin_time = trace_start_time
            # Plot the trace with times relative to origin time
            times_relative_to_origin = trace.times() + (trace_start_time - origin_time)
            ax.plot(
                times_relative_to_origin, trace.data, "k-", linewidth=0.8, label="Raw"
            )

        # Set y-axis label
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Only show x-axis label on bottom plot
        if i == len(trace_labels) - 1:
            ax.set_xlabel("Time (s)", fontsize=10)

    # Add legend from the first axis that has traces
    for ax in axes:
        if len(ax.lines) > 1:  # More than just the trace line
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles, strict=False))
            fig.legend(
                by_label.values(), by_label.keys(), loc="upper right", fontsize="small"
            )
            break

    # Set title and adjust layout
    fig.suptitle(
        f"Arrival Times for Station S{station_num:04d}\n{subtitle}", fontsize=16
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Leave room for suptitle

    # Save the plot
    fig.savefig(
        f"{output_dir}/station_{station_num:04d}_arrivals.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def find_stations_with_traces(folder_path):
    # Define the allowed BX?.sem? endings
    allowed_endings = {"BXX.semd", "BXZ.semd", "BXY.semr", "BXY.semir", "BXY.semc"}

    # Create regex pattern for the filename
    # AA.S????.S2.BX?.sem?
    pattern = r"^AA\.S(\d{4})\.S2\.(BX[XZY]\.sem[drci]?)$"

    stations_found = set()
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return set()

    # Search through all files in the folder
    for file_path in folder.iterdir():
        if file_path.is_file():
            filename = file_path.name
            match = re.match(pattern, filename)

            if match:
                station_num = match.group(1)  # Extract the 4-digit station number
                bx_ending = match.group(2)  # Extract the BX?.sem? part

                # Check if the ending is one of our allowed patterns
                if bx_ending in allowed_endings:
                    stations_found.add(int(station_num))
                    print(f"Found: {filename} -> Station {station_num}")

    return stations_found


def plot_all_stations(subtitle="", apply_filter=False):
    """
    Plot arrivals for all stations.
    """
    for station_num in find_stations_with_traces(f"{output_dir}"):
        print(f"Plotting arrivals for station {station_num}...")
        plot_station_with_arrivals(
            station_num, subtitle=subtitle, apply_filter=apply_filter
        )


if __name__ == "__main__":
    plot_all_stations(subtitle="", apply_filter=False)
