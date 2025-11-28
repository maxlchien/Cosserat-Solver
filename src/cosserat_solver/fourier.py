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
- 'oversample_rate': int  # oversampling factor for intermediate frequency sampling

The time array returned is [t0, t0 + dt, t0 + 2*dt, ..., t0 + (N-1)*dt].

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

    Returns
    -------
    np.ndarray
        Array of angular frequencies (rad/s) from 0 to Nyquist frequency.
        Shape: (N_oversample // 2 + 1,) where N_oversample = N * oversample_rate

    Notes
    -----
    For a signal sampled at interval dt, the Nyquist angular frequency is π/dt.
    The frequency spacing is dω = 2π/(N_oversample * dt).
    """

    # get number of dt's in the window
    N = len(_oversampled_time_array(ft_params))
    dt = ft_params["dt"]
    oversample_rate = ft_params["oversample_rate"]
    dt_oversample = dt / oversample_rate

    freqs = np.fft.fftfreq(N, dt_oversample)

    return 2 * np.pi * freqs  # convert to rad / sec


def _oversampled_time_array(ft_params: dict) -> np.ndarray:
    """
    Generate the oversampled time array used for intermediate calculations.

    Parameters
    ----------
    ft_params : dict
        Dictionary containing:
        - 'N': int, number of output samples
        - 'dt': float, time spacing between samples
        - 'oversample_rate': int, oversampling factor for intermediate calculations
        - 't0': float (optional), start time of the result (default: 0.0)
        - 'window_start': float (optional), start time of the window (default: t0 - dt * (N*oversample_rate//2))

    Returns
    -------
    np.ndarray
        Array of oversampled time samples used for intermediate calculations.
        Shape: (N_oversample,)
        [window_start, window_start + dt, ..., window_start + (N_oversample-1)*dt]
    """
    window = ft_params.get("support_window")
    dt = ft_params["dt"]
    N = ft_params["N"]
    oversample_rate = ft_params["oversample_rate"]
    N_oversample = N * oversample_rate
    dt_oversample = dt / oversample_rate
    t0 = ft_params.get("t0", 0.0)
    max_t0 = t0
    min_tf = max_t0 + dt_oversample * (N_oversample - 1)
    if window is not None:
        window_start = window[0]
        window_end = window[1]
    else:
        window_start = t0
        window_end = t0 + dt_oversample * (N_oversample - 1)

    # ensure original window is contained in expanded window
    if window_start > max_t0:
        window_start = max_t0
    if window_end < min_tf:
        window_end = min_tf

    return np.arange(window_start, window_end + dt_oversample / 2, dt_oversample)
    # return window_start + np.arange(N_oversample) * dt_oversample


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
        - 'oversample_rate': int, oversampling factor (>= 1) for accuracy
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

    Oversampling (oversample_rate > 1) computes the transform at higher resolution
    then downsamples to the requested N points for improved accuracy.
    """
    not_impl_msg = "cont_irfft is not yet implemented"
    raise NotImplementedError(not_impl_msg)
    # N = ft_params["N"]
    # dt = ft_params["dt"]
    # oversample_rate = ft_params["oversample_rate"]

    # N_oversample = N * oversample_rate
    # N_padded = N_oversample * pad_factor

    # # Create modified ft_params for the padded grid (without modifying original)
    # ft_params_padded = {
    #     "N": N_padded,
    #     "dt": dt,
    #     "oversample_rate": 1,  # Already accounted for in N_padded
    # }

    # # Get frequency array for the padded grid
    # omega = _omega_array(ft_params_padded)

    # # Evaluate the continuous FT function at discrete frequencies
    # freq_samples = func(omega)

    # # Ensure freq_samples has at least 2 dimensions for consistent handling
    # if freq_samples.ndim == 1:
    #     freq_samples = freq_samples[:, np.newaxis]
    #     squeeze_output = True
    # else:
    #     squeeze_output = False

    # # freq_samples shape: (n_freqs, *spatial_dims)
    # spatial_shape = freq_samples.shape[1:]

    # # Scaling factor: discretizing (1/2π) ∫ f_hat(ω) exp(iωt) dω
    # # With Δω = 2π/(N_padded * dt):
    # # f(t_n) ≈ (1/2π) Σ f_hat(ω_k) exp(iω_k t_n) * (2π/(N_padded * dt))
    # # = (1/(N_padded * dt)) Σ f_hat(ω_k) exp(iω_k t_n)
    # # After irfft (norm='backward'), we get: Σ f_hat(ω_k) exp(...)
    # # So we need to multiply by: 1/(N_padded * dt) but also account for
    # # the fact that we're using the actual dt spacing, not the padded spacing
    # # scale_factor = (2 * np.pi) / (N_padded * dt)
    # scale_factor = 1 / dt

    # # Compute inverse FFT
    # time_signal_padded = np.fft.irfft(freq_samples, n=N_padded, axis=0, norm="backward")
    # time_signal_padded *= scale_factor

    # # Extract only the first N_oversample samples (forward time window)
    # # These correspond to times [0, dt, 2*dt, ..., (N_oversample-1)*dt]
    # time_signal_oversampled = time_signal_padded[:N_oversample, ...]

    # # Downsample if oversampling was used
    # if oversample_rate > 1:
    #     time_signal = time_signal_oversampled[::oversample_rate, ...]
    # else:
    #     time_signal = time_signal_oversampled

    # # Restore original shape if input was 1D
    # if squeeze_output:
    #     time_signal = time_signal.squeeze(axis=1)

    # # Get time array
    # time = _time_array(ft_params)

    # return time, time_signal


def cont_ifft(
    f_hat: Callable[[float], complex] | Callable[[np.ndarray], np.ndarray],
    ft_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute inverse Fourier transform of a continuous frequency-domain function.

    Given a function that outputs the continuous Fourier transform spectrum,
    this discretizes it appropriately and applies numpy's ifft to recover
    the time-domain signal sampled at forward times [0, dt, 2*dt, ..., (N-1)*dt].

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
        - 'oversample_rate': int, oversampling factor (>= 1) for accuracy
        - 't0': float (optional), start time of the result (default: 0.0)
        - 'window_start': float (optional), start time of the window (default: t0 - dt * (N*oversample_rate//2))
            currently this does not do anything and is overriden by the default behavior.

    Returns
    -------
    time : np.ndarray
        Time array with shape (N,), values [0, dt, 2*dt, ..., (N-1)*dt]
    signal : np.ndarray
        Time-domain signal with shape (N, *func_shape), where func_shape is the
        shape of the output from func (excluding the frequency dimension)."""

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
    scale_factor = 1 / (ft_params["dt"] / ft_params["oversample_rate"])
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
