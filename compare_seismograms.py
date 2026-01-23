from __future__ import annotations

import argparse
import os
from argparse import RawTextHelpFormatter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

SUPPORTED_CHANNELS_WITH_EXTENSIONS = {
    "X": "BXX.semd",  # x displacement
    "Z": "BXZ.semd",  # z displacement
    "R": "BXY.semr",  # total rotation
    # "IR": "BXY.semir", # intrinsic rotation, not supported
    # "C": "BXY.semc", # curl, not supported
}


def load_trace(filepath):
    """
    Load seismogram trace from ASCII file.

    Parameters:
        filepath (str): Path to the trace file

    Returns:
        tuple: (time_array, displacement_array)
    """
    try:
        data = np.loadtxt(filepath)
        time = data[:, 0]
        displacement = data[:, 1]
        return time, displacement
    except Exception as e:
        err = f"Error loading trace from {filepath}: {e}"
        raise ValueError(err) from e


def get_sampling_rate(time_array):
    """
    Calculate sampling rate from time array.

    Parameters:
        time_array (np.array): Time values

    Returns:
        float: Sampling rate (samples per unit time)
    """
    dt = np.mean(np.diff(time_array))
    return 1.0 / dt


def find_common_time_range(time1, time2):
    """
    Find the overlapping time range between two traces.

    Parameters:
        time1, time2 (np.array): Time arrays for both traces

    Returns:
        tuple: (start_time, end_time) of common range
    """
    start_time = max(time1[0], time2[0])
    end_time = min(time1[-1], time2[-1])
    return start_time, end_time


def resample_to_common_grid(time1, disp1, time2, disp2, method="cubic"):
    """
    Resample both traces to a common time grid using interpolation.

    Parameters:
        time1, disp1 (np.array): First trace
        time2, disp2 (np.array): Second trace
        method (str): Interpolation method ('linear', 'cubic', 'quadratic')

    Returns:
        tuple: (common_time, resampled_disp1, resampled_disp2)
    """
    # Find common time range
    start_time, end_time = find_common_time_range(time1, time2)

    # Determine the finer sampling rate (higher resolution)
    sr1 = get_sampling_rate(time1)
    sr2 = get_sampling_rate(time2)
    target_sr = max(sr1, sr2)  # Use the higher sampling rate

    # Create common time grid
    dt = 1.0 / target_sr
    common_time = np.arange(start_time, end_time + dt / 2, dt)

    # Create interpolation functions
    f1 = interp1d(time1, disp1, kind=method, bounds_error=False, fill_value=0.0)
    f2 = interp1d(time2, disp2, kind=method, bounds_error=False, fill_value=0.0)

    # Resample both traces
    resampled_disp1 = f1(common_time)
    resampled_disp2 = f2(common_time)

    # Remove any NaN values that might have been introduced
    valid_idx = ~(np.isnan(resampled_disp1) | np.isnan(resampled_disp2))

    return (
        common_time[valid_idx],
        resampled_disp1[valid_idx],
        resampled_disp2[valid_idx],
    )


def get_compatible_traces(time1, disp1, time2, disp2, method="cubic"):
    """
    Resample both traces to a common time grid, using interpolation if necessary.

    Parameters:
        time1, disp1 (np.array): First trace
        time2, disp2 (np.array): Second trace
        method (str): Interpolation method ('linear', 'cubic', 'quadratic')

    Returns:
        tuple: (common_time, resampled_disp1, resampled_disp2)
    """
    # Find common time range
    start_time, end_time = find_common_time_range(time1, time2)
    time1_mask = (time1 >= start_time - 1e-6) & (time1 <= end_time + 1e-6)
    time2_mask = (time2 >= start_time - 1e-6) & (time2 <= end_time + 1e-6)
    time1_clipped = time1[time1_mask]
    time2_clipped = time2[time2_mask]

    # if the time grids are already the same, then don't interpolate to preserve data
    if len(time1_clipped) == len(time2_clipped) and np.all(
        np.isclose(time1_clipped, time2_clipped, rtol=1e-5, atol=1e-6)
    ):
        print("Time grids are close. Clipping data and skipping interpolation step.")
        return time1_clipped, disp1[time1_mask], disp2[time2_mask]
    print("Time grids are not close. Interpolating onto common grid.")
    return resample_to_common_grid(time1, disp1, time2, disp2, method)


def compute_fit_metrics(disp1, disp2):
    """
    Compute various fit metrics between two displacement traces.

    Parameters:
        disp1, disp2 (np.array): Displacement arrays to compare

    Returns:
        dict: Dictionary containing various fit metrics
    """
    # Remove any remaining NaN or infinite values
    valid_mask = np.isfinite(disp1) & np.isfinite(disp2)
    d1, d2 = disp1[valid_mask], disp2[valid_mask]

    if len(d1) == 0:
        return {"error": "No valid data points for comparison"}

    # Correlation coefficient
    correlation, p_value = pearsonr(d1, d2)

    # Mean squared error
    mse = np.mean((d1 - d2) ** 2)

    # Root mean squared error
    rmse = np.sqrt(mse)

    # Normalized RMSE (by range of reference trace)
    range_ref = np.max(d1) - np.min(d1)
    nrmse = rmse / range_ref if range_ref > 0 else np.inf

    # Mean absolute error
    mae = np.mean(np.abs(d1 - d2))

    # R-squared (coefficient of determination)
    ss_res = np.sum((d1 - d2) ** 2)
    ss_tot = np.sum((d1 - np.mean(d1)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Maximum absolute error
    max_error = np.max(np.abs(d1 - d2))

    return {
        "correlation": correlation,
        "correlation_p_value": p_value,
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "r_squared": r_squared,
        "max_error": max_error,
        "n_points": len(d1),
    }


def analyze_difference_spectrum(time, diff_signal):
    """
    Analyze the frequency content of the difference signal.

    Parameters:
        time (np.array): Time array
        diff_signal (np.array): Difference between traces

    Returns:
        dict: Dictionary containing frequency analysis results
    """
    # Calculate sampling rate
    dt = np.mean(np.diff(time))  # Use mean in case of slight irregularities
    fs = 1.0 / dt

    # Compute FFT
    N = len(diff_signal)
    fft_vals = np.fft.fft(diff_signal)
    freqs = np.fft.fftfreq(N, dt)

    # Take only positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    fft_pos = fft_vals[pos_mask]

    # Calculate power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_pos) ** 2

    return {
        "freqs": freqs_pos,
        "power": power_spectrum,
        "sampling_rate": fs,
        "freq_resolution": freqs_pos[1] - freqs_pos[0] if len(freqs_pos) > 1 else 0,
        "max_freq": freqs_pos[-1],
        "peak_freq": freqs_pos[np.argmax(power_spectrum)]
        if len(power_spectrum) > 0
        else 0,
        "peak_power": np.max(power_spectrum) if len(power_spectrum) > 0 else 0,
    }


def plot_individual_comparison(
    time,
    disp1,
    disp2,
    metrics,
    show_spectrum=True,
    title="Trace Comparison",
    save_to: str | Path = "trace_comparison.png",
):
    """
    Plot both traces for visual comparison.

    Parameters:
        time (np.array): Common time array
        disp1 (np.array): Displacement arrays
        disp2 (np.array): Displacement arrays
        title (str): Plot title
    """
    if show_spectrum:
        fig = plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot both traces
    ax1.plot(time, disp1, "b-", label="Reference Trace", linewidth=1.5)
    ax1.plot(time, disp2, "r--", label="Comparison Trace", linewidth=1.5, alpha=0.8)
    ax1.set_ylabel("Displacement")
    ax1.set_title("Traces")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot difference
    diff = disp1 - disp2
    ax2.plot(time, diff, "g-", linewidth=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Difference")
    ax2.set_title("Difference (Reference - Comparison)")
    ax2.grid(True, alpha=0.3)

    # Add frequency analysis if requested
    if show_spectrum:
        diff = disp1 - disp2
        spectrum_data = analyze_difference_spectrum(time, diff)

        ax3.semilogy(spectrum_data["freqs"], spectrum_data["power"])
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Power")
        ax3.set_title("Difference Signal Power Spectrum")
        ax3.grid(True, alpha=0.3)

        # Add spectrum info as text
        spectrum_text = f"Peak at {spectrum_data['peak_freq']:.1f} Hz\n"
        spectrum_text += f"Sampling rate: {spectrum_data['sampling_rate']:.1f} Hz"
        ax3.text(
            0.98,
            0.98,
            spectrum_text,
            transform=ax3.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            fontsize=9,
        )

    # add a box with the comparison metrics
    if "error" not in metrics:
        metrics_text = f"Correlation: {metrics['correlation']:.4f}\n"
        metrics_text += f"R²: {metrics['r_squared']:.4f}\n"
        metrics_text += f"RMSE: {metrics['rmse']:.2e}\n"
        metrics_text += f"NRMSE: {metrics['nrmse']:.4f}"

        ax1.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            fontsize=10,
        )
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def compare_individual_traces(
    filepath1,
    filepath2,
    method="cubic",
    plot=False,
    show_spectrum=True,
    plot_title: str = "Trace Comparison",
    output: str | Path = "trace_comparison.png",
) -> dict:
    """
    Main function to compare two seismogram traces.

    Parameters:
        filepath1 (str): Path to reference trace
        filepath2 (str): Path to comparison trace
        method (str): Interpolation method
        plot (bool): Whether to plot the comparison
        show_spectrum (bool): Whether to plot the spectrum.
        output (str, optional): The plot filepath. Defaults to "trace_comparison.png."

    Returns:
        dict: Fit metrics
    """
    # Load traces
    print("Loading traces...")
    time1, disp1 = load_trace(filepath1)
    time2, disp2 = load_trace(filepath2)

    print(
        f"Reference trace: {len(time1)} points, sampling rate: {get_sampling_rate(time1):.2f}"
    )
    print(
        f"Comparison trace: {len(time2)} points, sampling rate: {get_sampling_rate(time2):.2f}"
    )

    # Resample to common grid
    print(f"Resampling using {method} interpolation...")
    common_time, resampled_disp1, resampled_disp2 = get_compatible_traces(
        time1, disp1, time2, disp2, method=method
    )

    print(f"Common grid: {len(common_time)} points")
    print(f"Time range: {common_time[0]:.6f} to {common_time[-1]:.6f}")

    # Compute fit metrics
    print("Computing fit metrics...")
    metrics = compute_fit_metrics(resampled_disp1, resampled_disp2)

    # Plot if requested
    if plot:
        plot_individual_comparison(
            common_time,
            resampled_disp1,
            resampled_disp2,
            metrics,
            show_spectrum,
            title=plot_title,
            save_to=output,
        )

    return metrics


def run_individual_comparison_on_folder(
    folder: str,
    station_num: int,
    channel: str,
    method: str = "cubic",
    plot: bool = False,
    show_spectrum: bool = True,
    output: str | Path | None = None,
) -> dict:
    """
    Run an individual comparison on the specified channel and station.

    Parameters:
        folder (str): The folder to run comparisons. Needs to contain a `solver` and `specfem` folder, each of which contains traces in an `output` subfolder.
        station_num: Description
        channel: Description
    """
    base_path = Path(folder)
    if channel not in SUPPORTED_CHANNELS_WITH_EXTENSIONS:
        err = f"{channel} is not a supported key. Supported keys and corresponding extensions:\n{SUPPORTED_CHANNELS_WITH_EXTENSIONS}"
        raise ValueError(err)
    ext = SUPPORTED_CHANNELS_WITH_EXTENSIONS[channel]
    specfem_path = (
        base_path / "specfem" / "output" / f"AA.S{str(station_num).zfill(4)}.S2.{ext}"
    )
    solver_path = (
        base_path / "solver" / "output" / f"AA.S{str(station_num).zfill(4)}.S2.{ext}"
    )
    if not specfem_path.is_file():
        err = f"Requested channel {channel} is not available for SPECFEM station {station_num} at filename {specfem_path}"
        raise ValueError(err)
    if not solver_path.is_file():
        err = f"Requested channel {channel} is not available for solver station {station_num} at filename {solver_path}"
        raise ValueError(err)
    if output is None:
        output = base_path / f"station_{station_num}_comparison_{channel}.png"
    else:
        output = base_path / output
    return compare_individual_traces(
        specfem_path,
        solver_path,
        method,
        plot,
        show_spectrum,
        plot_title=f"Trace Comparison Analysis for Channel {channel} of Station {station_num}",
        output=output,
    )


def compare_at_station_level(
    folder: str,
    station_num: int,
    method: str = "cubic",
    plot: bool = False,
    show_spectrum: bool = True,
    output: str | Path | None = None,
) -> None:
    """
    Run a folder level, all-channels comparison for a single station.

    Parameters:
        folder (str): The folder to run comparisons. Needs to contain a `solver` and `specfem` folder, each of which contains traces in an `output` subfolder.
        station_num: Description
        channel: Description
    """
    base_path = Path(folder)
    channels_present = []
    specfem_channels = []
    solver_channels = []
    # check which channels are in both the specfem and solver output
    for short, ext in SUPPORTED_CHANNELS_WITH_EXTENSIONS.items():
        specfem_path = (
            base_path
            / "specfem"
            / "output"
            / f"AA.S{str(station_num).zfill(4)}.S2.{ext}"
        )
        solver_path = (
            base_path
            / "solver"
            / "output"
            / f"AA.S{str(station_num).zfill(4)}.S2.{ext}"
        )
        if specfem_path.is_file():
            specfem_channels.append(short)
        if solver_path.is_file():
            solver_channels.append(short)
        if specfem_path.is_file() and solver_path.is_file():
            channels_present.append(short)
    if not channels_present:
        err = f"No channels are shared between specfem and solver input.\nSpecfem channels: {specfem_channels}\nSolver channels: {solver_channels}"
        raise Exception(err)
    if not output:
        output = f"station_{station_num}_comparison.png"
    output = base_path / Path(output)

    print("=" * 50)
    stored_traces = {}
    for channel_short in channels_present:
        ext = SUPPORTED_CHANNELS_WITH_EXTENSIONS[channel_short]
        print(f"Running comparison for channel {channel_short}")
        specfem_path = (
            base_path
            / "specfem"
            / "output"
            / f"AA.S{str(station_num).zfill(4)}.S2.{ext}"
        )
        solver_path = (
            base_path
            / "solver"
            / "output"
            / f"AA.S{str(station_num).zfill(4)}.S2.{ext}"
        )
        time1, disp1 = load_trace(specfem_path)
        time2, disp2 = load_trace(solver_path)

        print(
            f"Specfem trace: {len(time1)} points, sampling rate: {get_sampling_rate(time1):.2f}"
        )
        print(
            f"Solver trace: {len(time2)} points, sampling rate: {get_sampling_rate(time2):.2f}"
        )

        # Resample to common grid
        common_time, resampled_disp1, resampled_disp2 = get_compatible_traces(
            time1, disp1, time2, disp2, method=method
        )

        print(f"Common grid: {len(common_time)} points")
        print(f"Time range: {common_time[0]:.6f} to {common_time[-1]:.6f}")

        # Compute fit metrics
        metrics = compute_fit_metrics(resampled_disp1, resampled_disp2)
        print_metrics(metrics)

        stored_traces[channel_short] = (
            common_time,
            resampled_disp1,
            resampled_disp2,
            metrics,
        )

        if plot:
            plot_individual_comparison(
                common_time,
                resampled_disp1,
                resampled_disp2,
                metrics,
                show_spectrum,
                title=f"Trace Comparison Analysis for Channel {channel_short} of Station {station_num}",
                save_to=output.with_stem(f"{output.stem}_{channel_short}"),
            )
        print("=" * 50)

    if plot:
        n = 3 * len(channels_present) if show_spectrum else 2 * len(channels_present)
        fig, axes = plt.subplots(n, figsize=(12, 4 * n))
        i = -1
        for channel_short in channels_present:
            # Plot both traces
            time, disp1, disp2, metrics = stored_traces[channel_short]
            i += 1
            ax = axes[i]
            ax.plot(time, disp1, "b-", label="Specfem Trace", linewidth=1.5)
            ax.plot(time, disp2, "r--", label="Solver Trace", linewidth=1.5, alpha=0.8)
            ax.set_ylabel("Displacement")
            ax.set_title(f"Trace Comparison for Channel {channel_short}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # add a box with the comparison metrics
            if "error" not in metrics:
                metrics_text = f"Correlation: {metrics['correlation']:.4f}\n"
                metrics_text += f"R²: {metrics['r_squared']:.4f}\n"
                metrics_text += f"RMSE: {metrics['rmse']:.2e}\n"
                metrics_text += f"NRMSE: {metrics['nrmse']:.4f}"

                ax.text(
                    0.02,
                    0.98,
                    metrics_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                    fontsize=10,
                )

            # Plot difference
            i += 1
            ax = axes[i]
            diff = disp1 - disp2
            ax.plot(time, diff, "g-", linewidth=1)
            ax.set_xlabel("Time")
            ax.set_ylabel("Difference")
            ax.set_title(f"Difference (Specfem - Solver) for Channel {channel_short}")
            ax.grid(True, alpha=0.3)

            # Add frequency analysis if requested
            if show_spectrum:
                i += 1
                ax = axes[i]
                spectrum_data = analyze_difference_spectrum(time, diff)

                ax.semilogy(spectrum_data["freqs"], spectrum_data["power"])
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Power")
                ax.set_title(
                    f"Difference Signal Power Spectrum for Channel {channel_short}"
                )
                ax.grid(True, alpha=0.3)

                # Add spectrum info as text
                spectrum_text = f"Peak at {spectrum_data['peak_freq']:.1f} Hz\n"
                spectrum_text += (
                    f"Sampling rate: {spectrum_data['sampling_rate']:.1f} Hz"
                )
                ax.text(
                    0.98,
                    0.98,
                    spectrum_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                    fontsize=9,
                )
        fig.suptitle(f"Trace Comparison Analysis for Station {station_num}")

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(output)
        plt.close()


def print_metrics(metrics):
    """Print fit metrics in a readable format."""
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print("\n" + "=" * 50)
    print("FIT METRICS")
    print("=" * 50)
    print(f"Number of comparison points: {metrics['n_points']:,}")
    print(f"Correlation coefficient:     {metrics['correlation']:.6f}")
    print(f"Correlation p-value:         {metrics['correlation_p_value']:.2e}")
    print(f"R-squared:                   {metrics['r_squared']:.6f}")
    print(f"Mean Squared Error (MSE):    {metrics['mse']:.2e}")
    print(f"Root Mean Squared Error:     {metrics['rmse']:.2e}")
    print(f"Normalized RMSE:             {metrics['nrmse']:.6f}")
    print(f"Mean Absolute Error:         {metrics['mae']:.2e}")
    print(f"Maximum Absolute Error:      {metrics['max_error']:.2e}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two seismogram traces, possibly with different sampling rates."
        "\n"
        "This program runs in three main modes:"
        f'\n- Automatic selection mode, one channel: Specify a folder with --folder, which must have "specfem/output" and "solver/output" subfolders containing traces. Specify --station (int) and --channel. Channel options and corresponding extensions are {SUPPORTED_CHANNELS_WITH_EXTENSIONS}.'
        "\n- Automatic selection mode, all channels: Specify a folder and station number as in the one channel mode, but leave --channel unspecified. Generates station-level visualizations. Optional specify --all-stations to run this for each station present in the folder."
        "\n- Manual selection mode: Specify two filepaths --trace1 and --trace2 pointing to ascii format trace data. Specify an output filepath --output.",
        formatter_class=RawTextHelpFormatter,
    )
    automatic_arguments = parser.add_argument_group("Automatic Selection Mode")
    automatic_arguments.add_argument(
        "--folder", help="Folder for comparing (as a subfolder of comparisons folder)"
    )
    automatic_arguments.add_argument(
        "--external-folder",
        dest="external_folder",
        action="store_true",
        help="Flag for specifying that the argument for --folder is not a subfolder of comparisons/",
    )
    automatic_arguments.add_argument(
        "--station", default=1, help="Station number for comparison. Defaults to 1."
    )
    automatic_arguments.add_argument(
        "--channel",
        choices=SUPPORTED_CHANNELS_WITH_EXTENSIONS.keys(),
        help="Channel of trace for individual comparison (must exist in both solver and specfem folders),\nIf not set then defaults to broad comparison.",
    )
    automatic_arguments.add_argument(
        "--all-stations",
        dest="all_stations",
        action="store_true",
        help="Run station comparison for all stations in a folder. Only available in automatic selection mode at the folder level.",
    )

    # manual selection mode
    manual_arguments = parser.add_argument_group("Manual Selection Mode")
    manual_arguments.add_argument(
        "--trace1", help="Filepath to first trace for manual trace comparison"
    )
    manual_arguments.add_argument(
        "--trace2", help="Filepath to second trace for manual trace comparison"
    )

    general_options = parser.add_argument_group("General Options")
    general_options.add_argument(
        "--output",
        help="Filepath to output image. Only used if --plot is selected. In automatic mode, interpreted as a path relative to the folder. In manual and one channel mode, the individual comparison is plotted here. In all channel (single station mode), the station comparison is plotted here, and individual comparisons are plotted at the same path, with the channel as the extension. Does nothing when all stations are enabled.",
    )
    general_options.add_argument(
        "--method",
        default="cubic",
        choices=["linear", "cubic", "quadratic"],
        help="Interpolation method (default: cubic)",
    )
    general_options.add_argument(
        "--plot", action="store_true", help="Show comparison plot"
    )
    general_options.add_argument(
        "--spectrum",
        action="store_true",
        help="Show frequency spectrum of difference signal",
    )

    args = parser.parse_args()

    ### MANUAL TRACE SELECTION
    if args.trace1 or args.trace2:
        # check if traces are manually enabled:
        if not (args.trace1 and args.trace2):
            err = "Error: both --trace1 and --trace2 must be specified for manual trace selection."
            raise ValueError(err)
        if args.channel:
            err = "Channel cannot be automatically selected if using manual trace selection."
            raise ValueError(err)
        # run the manual trace selection from here
        print("=" * 50)
        print(
            f"Running comparison on manually selected traces at paths {args.trace1} and {args.trace2}."
        )

        if args.all_stations:
            err = "--all-stations flag is incompatible with manual selection mode."
            raise ValueError(err)

        try:
            metrics = compare_individual_traces(
                args.trace1,
                args.trace2,
                args.method,
                args.plot,
                args.spectrum,
                output=args.output,
            )

            print_metrics(metrics)
        except Exception as e:
            print(f"Error during comparison: {e}")

    ### AUTOMATIC TRACE SELECTION
    if not args.folder:
        err = "Either --folder (automatic trace selection) or --trace1 and --trace2 (manual trace selection) must be specified."
        raise ValueError(err)
    if not args.station:
        err = "For automatic selection mode, a station number must be specified."
        raise ValueError(err)
    folder_path = args.folder if args.external_folder else "comparisons/" + args.folder
    if not os.path.exists(folder_path):
        err = f"Specified folder {folder_path} does not exist."
        raise ValueError(err)
    print("=" * 50)
    print(f"Proceeding with automatic trace selection at folder {folder_path}")
    if args.channel:
        # individual comparison

        if args.all_stations:
            err = "--all-stations flag is incompatible with single channel mode."
            raise ValueError(err)
        try:
            # Compare individual trace if requested
            metrics = run_individual_comparison_on_folder(
                folder_path,
                args.station,
                args.channel,
                args.method,
                args.plot,
                args.spectrum,
                output=args.output,
            )

            # Print results
            print_metrics(metrics)

        except Exception as e:
            print(f"Error during comparison: {e}")
    else:
        # TODO: get --all-stations working
        if args.all_stations:
            err = "--all-stations is not yet implemented."
            raise NotImplementedError(err)
        compare_at_station_level(
            folder_path,
            args.station,
            args.method,
            args.plot,
            args.spectrum,
            output=args.output,
        )
        # whole comparison


if __name__ == "__main__":
    main()
