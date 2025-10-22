from __future__ import annotations

import numpy as np

"""
    Compute the inverse Fourier transform of a np.ndarray valued function of omega.

    Parameters:
    func: Callable
        The function to be transformed. This should be a function of omega which returns a np.ndarray. For instance, func may
        be the Green's function in the frequency domain multiplied pointwise by the source spectrum.
    ft_params: dict
            A dictionary containing the parameters for the Fourier transform. It should contain the following keys:
            - 'dt': The time step size for the output trace.
            - 'N': The number of time samples for the output trace.
            - 'oversample_rate': An integer specifying the oversampling rate for internal frequency sampling.
"""


def _omega_array(ft_params: dict) -> np.ndarray:
    """
    Create an array of frequencies for the Fourier transform.

    Parameters:
    ft_params: dict
        A dictionary containing the parameters for the Fourier transform. It should contain the following keys:
        - 'dt': The time step size for the output trace.
        - 'N': The number of time samples for the output trace.
        - 'oversample_rate': An integer specifying the oversampling rate for internal frequency sampling.
    Returns:
    np.ndarray
        An array of frequencies in radians per second. The length will be the next power of 2 greater than or equal to 'num_freqs' for efficiency.
    """
    dt = ft_params.get("dt")
    N = ft_params.get("N")
    oversample_rate = ft_params.get("oversample_rate", 1)
    if not isinstance(oversample_rate, int) or oversample_rate < 1:
        msg = f"oversample_rate must be a positive integer, got {oversample_rate}"
        raise ValueError(msg)

    dt_internal = dt / oversample_rate
    N_internal = N * oversample_rate
    N_pow2 = 2 ** int(np.ceil(np.log2(N_internal)))  # Next power of 2 for efficiency

    freqs_hz = np.fft.rfftfreq(N_pow2, d=dt_internal)

    return 2 * np.pi * freqs_hz  # Convert to radians per second


def itransform(func, ft_params: dict) -> np.ndarray:
    """
    Compute the inverse Fourier transform of the function.

    Returns:
    np.ndarray
        The time-domain signal obtained by inverse Fourier transforming the input function.
    """
    omegas = _omega_array(ft_params)

    omega_trace = np.zeros((len(omegas), *func(omegas[0]).shape), dtype=np.complex128)
    time_trace = np.zeros((len(omegas), *func(omegas[0]).shape), dtype=np.complex128)

    for i, omega in enumerate(omegas):
        if (i % 100) == 0:
            print(f"Computing frequency {i + 1} / {len(omegas)}: omega={omega}")
        omega_trace[i] = func(omega)

    # take irfft component by component
    for i in range(omega_trace.shape[1]):
        for j in range(omega_trace.shape[2]):
            time_trace[:, i, j] = np.fft.irfft(omega_trace[:, i, j], n=len(omegas))

    # remove the oversampled points and truncate back to requested number
    return time_trace[:: ft_params.get("oversample_rate", 1)][: ft_params.get("N")]
