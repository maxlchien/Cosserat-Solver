from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr


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
        raise ValueError(f"Error loading trace from {filepath}: {e}")


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


def plot_comparison(
    time, disp1, disp2, metrics, show_spectrum=True, title="Trace Comparison"
):
    """
    Plot both traces for visual comparison.

    Parameters:
    time (np.array): Common time array
    disp1, disp2 (np.array): Displacement arrays
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
    ax1.set_title(title)
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
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
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
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig("trace_comparison.png")
    plt.close()


def compare_individual_traces(
    filepath1, filepath2, method="cubic", plot=False, show_spectrum=True
):
    """
    Main function to compare two seismogram traces.

    Parameters:
    filepath1 (str): Path to reference trace
    filepath2 (str): Path to comparison trace
    method (str): Interpolation method
    plot (bool): Whether to plot the comparison

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
    common_time, resampled_disp1, resampled_disp2 = resample_to_common_grid(
        time1, disp1, time2, disp2, method=method
    )

    print(f"Common grid: {len(common_time)} points")
    print(f"Time range: {common_time[0]:.6f} to {common_time[-1]:.6f}")

    # Compute fit metrics
    print("Computing fit metrics...")
    metrics = compute_fit_metrics(resampled_disp1, resampled_disp2)

    # Plot if requested
    if plot:
        plot_comparison(
            common_time, resampled_disp1, resampled_disp2, metrics, show_spectrum
        )

    return metrics


def run_individual_comparison_on_folder(folder: str, station_num: int, channel: str):
    """
    Run an individual comparison on the specified channel and station.

    Parameters:
        folder (str): The folder to run comparisons. Needs to contain a `solver` and `specfem` folder, each of which contains traces in an `output` subfolder.
        station_num: Description
        channel: Description
    """


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
        description="Compare two seismogram traces with different sampling rates"
    )
    parser.add_argument(
        "folder", help="Folder for comparing (as a subfolder of comparisons folder)"
    )
    parser.add_argument(
        "station", default=1, help="Station number for comparison. Defaults to 1."
    )
    parser.add_argument(
        "--channel",
        choices=[
            "BXX",
            "BXY",
            "BXZ",
        ],
        help="Channel of trace for individual comparison (must exist in both solver and specfem folders),\nIf not set then defaults to broad comparison.",
    )
    parser.add_argument(
        "--method",
        default="cubic",
        choices=["linear", "cubic", "quadratic"],
        help="Interpolation method (default: cubic)",
    )
    parser.add_argument("--plot", action="store_true", help="Show comparison plot")
    parser.add_argument(
        "--spectrum",
        action="store_true",
        help="Show frequency spectrum of difference signal",
    )

    args = parser.parse_args()
    if args.trace:
        # individual comparison
        trace1 = f"comparisons/{args.folder}/specfem/{args.trace}"
        trace2 = f"cosserat_traces/{args.trace}"
        # Check if files exist
        for filepath in [trace1, trace2]:
            if not os.path.exists(filepath):
                print(f"Error: File {filepath} not found")
                return

        try:
            # Compare individual trace if requested
            metrics = compare_individual_traces(
                trace1, trace2, args.method, args.plot, args.spectrum
            )

            # Print results
            print_metrics(metrics)

        except Exception as e:
            print(f"Error during comparison: {e}")
    else:
        ...
        # whole comparison


if __name__ == "__main__":
    main()
