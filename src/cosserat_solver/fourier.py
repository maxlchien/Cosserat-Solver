"""
Wraps the discrete Fourier transform np.ifft to invert continuous Fourier transforms.

If f(t), f_hat(omega) are a continuous (complex valued, not np.ndarray) FT pair with conventions:
f_hat(omega) = int f(t) exp(+i omega t) dt

then cont_ifft(f_hat) provided in this module returns time, signal, where
signal = f(time)

The time array is specified in a dict ft_params, which contains:
- 'N' : int            # number of time samples returned
- 'dt' : float         # time spacing (seconds)
- 't0' : float = 0.0, optional   # time-shift applied to result
- 'support_window': tuple[float, float], optional  # (t_start, t_end) of the window where f(t) is supported
    if not provided then defaults to (t0, t0 + dt * (N-1))
- 'extension_factor': int  # factor for extending the window [t0, t0+N*dt] to [t0, t0+N*dt*extension_factor]
- 'refinement_factor': int, optional  # factor for finer frequency sampling (dt/refinement_factor spacing)
    Defaults to extension_factor if not provided

The time array returned is [t0, t0 + dt, t0 + 2*dt, ..., t0 + (N-1)*dt].
The frequency sampling is conducted over the extended window at refined resolution.

If f is real-valued, then cont_irfft can be used instead. NOTE: currently not implemented
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d


def _omega_array(ft_params: dict) -> np.ndarray:
    """
    Generate the angular frequency array for evaluating the continuous Fourier transform.

    This returns the non-negative angular frequencies corresponding to numpy's irfft,
    which expects frequencies from 0 to the Nyquist frequency.

    Parameters
    ----------
    ft_params : dict
        Dictionary containing:
        - 'N': int, number of output samples
        - 'dt': float, time spacing between samples
        - 'extension_factor': int, extension factor
        - 'refinement_factor': int (optional), refinement factor

    Returns
    -------
    np.ndarray
        Array of angular frequencies (rad/s).
        Shape: (N_oversample,) where N_oversample depends on refinement and extension factors

    Notes
    -----
    The frequency spacing is dω = 2π/(N_oversample * dt_refined) where
    dt_refined = dt / refinement_factor.
    """

    # get number of dt's in the window
    N = len(_oversampled_time_array(ft_params))
    dt = ft_params["dt"]
    refinement_factor = ft_params.get("refinement_factor", 1)
    dt_refined = dt / refinement_factor

    freqs = np.fft.fftfreq(N, dt_refined)

    return 2 * np.pi * freqs  # convert to rad / sec


def _oversampled_time_array(ft_params: dict) -> np.ndarray:
    """
    Generate the oversampled time array used for intermediate calculations.

    Supports both refinement (finer sampling within a window) and extension (expanding the window).
    These factors are applied independently:
    - refinement_factor: Controls sampling density (dt/refinement_factor)
    - extension_factor: Controls window extent ([t0, t0 + N*dt*extension_factor])

    Parameters
    ----------
    ft_params : dict
        Dictionary containing:
        - 'N': int, number of output samples
        - 'dt': float, time spacing between samples
        - 'extension_factor': int, factor for extending the window
        - 'refinement_factor': int (optional), factor for finer frequency sampling
          Defaults to 1 if not provided
        - 't0': float (optional), start time of the result (default: 0.0)
        - 'support_window': tuple[float, float] (optional), custom window (t_start, t_end)
          If provided, the computed window will be expanded to contain it.

    Returns
    -------
    np.ndarray
        Array of oversampled time samples used for intermediate calculations.
        Shape: (N_oversample,) where N_oversample depends on refinement and extension factors
        Spans the window [window_start, window_start + dt_refined, ..., window_start + (N_oversample-1)*dt_refined]
        where dt_refined = dt / refinement_factor and window spans [t0, t0 + N*dt*extension_factor]
    """
    window = ft_params.get("support_window")
    dt = ft_params["dt"]
    N = ft_params["N"]
    extension_factor = ft_params.get("extension_factor", 1)

    # Get refinement factor, defaulting to 1
    refinement_factor = ft_params.get("refinement_factor", 1)

    # The refined time step
    dt_refined = dt / refinement_factor

    # The window spans [t0, t0 + N*dt*extension_factor]
    t0 = ft_params.get("t0", 0.0)
    window_start = t0
    window_end = t0 + N * dt * extension_factor

    # If a custom support_window is provided, expand our computed window to contain it
    if window is not None:
        window_start = min(window_start, window[0])
        window_end = max(window_end, window[1])

    return np.arange(window_start, window_end + dt_refined / 2, dt_refined)


def downsample_signal(expanded_time, downsampled_time, expanded_signal) -> np.ndarray:
    """
    Downsample the expanded time-domain signal to match the downsampled time array.

    Parameters
    ----------
    expanded_time : np.ndarray
        The oversampled time array used for intermediate calculations.
        Shape: (N_oversample,)
    downsampled_time : np.ndarray
        The desired downsampled time array. Assumed to be a contiguous subset of expanded_time.
        Shape: (N,)
    expanded_signal : np.ndarray
        The oversampled time-domain signal.
        Shape: (N_oversample, *spatial_dims)

    Returns
    -------
    np.ndarray
        The downsampled time-domain signal.
        Shape: (N, *spatial_dims)
    """
    # downsample with linear interpolation
    interpolator = interp1d(
        expanded_time, expanded_signal, axis=0, kind="linear", fill_value="extrapolate"
    )
    return interpolator(downsampled_time)


def _time_array(ft_params: dict) -> np.ndarray:
    """
    Generate the time array corresponding to the output trace.

    Parameters
    ----------
    ft_params : dict
        Dictionary containing:
        - 'N': int, number of output samples
        - 'dt': float, time spacing between samples
        - 't0': float (optional), start time of the result (default: 0.0)

    Returns
    -------
    np.ndarray
        Array of time samples corresponding to the output trace.
        Shape: (N,)
        [t0, t0 + dt, t0 + 2*dt, ..., t0 + (N-1)*dt]
    """
    N = ft_params["N"]
    dt = ft_params["dt"]

    # the forward window 0, dt, 2*dt, ..., (N-1)*dt, without shifting
    t = np.arange(N) * dt

    return t + ft_params.get("t0", 0.0)


def cont_irfft(func, ft_params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    NOTE: currently not implemented. Use cont_ifft and take the real part instead.
    Compute inverse Fourier transform of a continuous frequency-domain function.

    Given a function that outputs the continuous Fourier transform spectrum,
    this discretizes it appropriately and applies numpy's irfft to recover
    the time-domain signal sampled at forward times [0, dt, 2*dt, ..., (N-1)*dt].

    Parameters
    ----------
    func : callable
        Function with signature func(omega: float) -> float
        Returns the continuous Fourier transform at angular frequency omega.
        Should be conjugate symmetric (func(-ω) = conj(func(ω))) for real signals.
    ft_params : dict
        Dictionary containing:
        - 'N': int, number of output samples in time domain
        - 'dt': float, time spacing between samples (seconds)
        - 'extension_factor': int, extension factor for extended window
        - 'refinement_factor': int (optional), refinement factor for finer sampling
        - 'pad_factor': int (optional), zero-padding factor to avoid wraparound (default: 2)

    Returns
    -------
    time : np.ndarray
        Time array with shape (N,), values [0, dt, 2*dt, ..., (N-1)*dt]
    signal : np.ndarray
        Time-domain signal with shape (N, *func_shape), where func_shape is the
        shape of the output from func (excluding the frequency dimension).

    Notes
    -----
    Conventions:
    - Forward FT: f_hat(ω) = ∫ f(t) exp(-iωt) dt
    - Inverse FT: f(t) = (1/2π) ∫ f_hat(ω) exp(iωt) dω

    The function uses zero-padding to avoid periodic wraparound artifacts. Signals
    with support at negative times will not contaminate the forward-time output
    window [0, (N-1)*dt].

    Refinement and extension improve accuracy by computing the transform at higher
    resolution then downsampling to the requested N points.
    """
    not_impl_msg = "cont_irfft is not yet implemented"
    raise NotImplementedError(not_impl_msg)


def cont_ifft(
    f_hat: Callable[[float], complex] | Callable[[np.ndarray], np.ndarray],
    ft_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute inverse Fourier transform of a continuous frequency-domain function.

    Given a function that outputs the continuous Fourier transform spectrum,
    this discretizes it appropriately and applies numpy's ifft to recover
    the time-domain signal.

    Parameters
    ----------
    f_hat : callable
        Function with signature f_hat(omega: float) -> complex or vectorized as f_hat(omega: np.ndarray) -> np.ndarray
        Returns the continuous Fourier transform at angular frequency omega.
        Should be conjugate symmetric (f_hat(-ω) = conj(f_hat(ω))) for real signals.
    ft_params : dict
        Dictionary containing:
        - 'N': int, number of output samples in time domain
        - 'dt': float, time spacing between samples (seconds)
        - 'extension_factor': int, factor for extending the output window [t0, t0 + N*dt*extension_factor]
        - 't0': float (optional), start time of the result (default: 0.0)
        - 'refinement_factor': int (optional), factor for finer frequency sampling
            The frequency samples are computed at spacing dt/refinement_factor.
            Defaults to 1 if not provided.
        - 'support_window': tuple[float, float] (optional), custom window bounds
            This expands the computed window to contain it.

    Returns
    -------
    time : np.ndarray
        Time array corresponding to the output trace, shape (N,), values [t0, t0 + dt, ..., t0 + (N-1)*dt]
    signal : np.ndarray
        Time-domain signal with shape (N, *func_shape), where func_shape is the
        shape of the output from func (excluding the frequency dimension).

    Notes
    -----
    The refinement_factor and extension_factor are applied independently:
    - refinement_factor controls how finely the frequency domain is sampled (defaults to 1)
    - extension_factor controls how much to extend the time window
    The frequency domain is sampled densely over the extended window, then the
    time-domain result is downsampled to the output time grid.
    """

    expanded_time = _oversampled_time_array(ft_params)
    omega_array = _omega_array(ft_params)

    try:
        # vectorized
        freq_samples = f_hat(omega_array)
    except Exception:
        # nonvectorized
        freq_samples = np.array([f_hat(omega) for omega in omega_array])

    freq_samples = np.conj(freq_samples)  # conjugate to match fourier convention

    # phase shift
    shifted_time = -(expanded_time[0] + expanded_time[-1]) / 2
    phase_shift = np.exp(-1j * omega_array * shifted_time)
    # Reshape scalars to have the right number of dimensions
    num_extra_dims = freq_samples.ndim - 1
    shape = (len(phase_shift),) + (1,) * num_extra_dims
    freq_samples_shifted = freq_samples * phase_shift.reshape(shape)

    expanded_time_signal = np.fft.ifft(freq_samples_shifted, axis=0, norm="backward")

    # fix to account for conventions
    refinement_factor = ft_params.get("refinement_factor", 1)
    scale_factor = 1 / (ft_params["dt"] / refinement_factor)
    expanded_time_signal *= scale_factor
    expanded_time_signal = np.conj(
        expanded_time_signal
    )  # conjugate back in case signal was complex

    expanded_time_signal = np.fft.fftshift(expanded_time_signal, axes=0)

    # extract forward time window
    final_signal = downsample_signal(
        expanded_time, _time_array(ft_params), expanded_time_signal
    )

    return _time_array(ft_params), final_signal
