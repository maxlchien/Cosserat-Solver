from __future__ import annotations

import numpy as np
import pytest

from cosserat_solver.fourier import cont_ifft, downsample_signal


def gaussian_pair(t0=0.0, sigma=1.0):
    r"""
    Gaussian pulse centered at t0 with width sigma.

    Forward FT convention: f_hat(ω) = ∫ f(t) exp(-iωt) dt

    f(t) = exp(-(t-t0)^2/(2\alpha^2))

    f_hat(\omega) = \sigma√(2π) exp(-i\omega t0) exp(-\sigma^2\omega^2/2)
    """

    def f(t):
        return np.exp(-((t - t0) ** 2) / (2 * sigma**2))

    def fhat(omega):
        return (
            sigma
            * np.sqrt(2 * np.pi)
            * np.exp(1j * omega * t0)
            * np.exp(-(sigma**2) * omega**2 / 2)
        )

    support_window = (t0 - 15 * sigma, t0 + 15 * sigma)
    return f, fhat, support_window


def ricker_pair(f0=1.0, tshift=0.0):
    r"""
    Ricker wavelet (Mexican hat wavelet).

    f(t) = (1 - 2(\pi f0 t)^2) exp(-(\pi f0 t)^2)
    f_hat(\omega) = (1 / (2 \pi^{5/2} f0^3)) \omega^2 \exp(-\omega^2 / (4 (\pi f0)^2)) \exp(i \omega tshift)
    """

    def f(t):
        return (1 - 2 * (np.pi * f0 * (t - tshift)) ** 2) * np.exp(
            -((np.pi * f0 * (t - tshift)) ** 2)
        )

    def fhat(omega):
        return (
            (1 / (2 * np.pi ** (5 / 2) * f0**3))
            * omega**2
            * np.exp(-(omega**2) / (4 * (np.pi * f0) ** 2))
            * np.exp(1j * omega * tshift)
        )

    support_window = (tshift - 100 / f0, tshift + 100 / f0)
    return f, fhat, support_window


def raised_cosine_pair(T=1.0, t0=0.0):
    r"""
    Raised cosine pulse (smooth, narrowly supported).

    f(t) = (1 + cos(π(t-t0)/T)) / 2  for |t-t0| ≤ T, 0 otherwise

    This is narrowly supported in time over [t0-T, t0+T] and has smooth
    boundaries due to the cosine taper.

    For the forward Fourier convention f_hat(ω) = ∫ f(t) exp(iωt) dt:

    Since f is even and real, f_hat is real and even. The transform is:
    f_hat(ω) = (π/T)² * sin(ωT) / (ωT * ((π/T)² - ω²)) * exp(iωt0)

    Equivalently: f_hat(ω) = sinc(ωT/π) * [1 + cos(ωT)] where sinc(x) = sin(πx)/(πx)
    """

    def f(t):
        t_shifted = t - t0
        mask = np.abs(t_shifted) <= T
        return np.where(mask, (1 + np.cos(np.pi * t_shifted / T)) / 2, 0.0)

    def fhat(omega):
        """
        Analytical Fourier transform of raised cosine.

        Convention: f_hat(ω) = ∫ f(t) exp(iωt) dt

        For a real, even function, the FT is real.
        f_hat(ω) = (π/T)² * sin(ωT) / (ωT * ((π/T)² - ω²)) * exp(iωt0)
        """
        # Convert scalar to array for uniform handling
        is_scalar = not isinstance(omega, np.ndarray)
        omega_arr = np.atleast_1d(omega)

        result = np.zeros(len(omega_arr), dtype=complex)
        exp_shift = np.exp(1j * omega_arr * t0)

        eps = 1e-10

        for i, w in enumerate(omega_arr):
            if abs(w) < eps:
                # Limit as ω->0
                result[i] = 1.0 * exp_shift[i]
            elif abs((w * T) ** 2 - np.pi**2) < eps:
                result[i] = np.nan
            else:
                # General formula: (π/T)² * sin(ωT) / (ωT * ((π/T)² - ω²))
                pi_over_T = np.pi / T
                numerator = pi_over_T**2 * np.sin(w * T)
                denominator = w * T * (pi_over_T**2 - w**2)
                result[i] = (numerator / denominator) * exp_shift[i]

        return result[0] if is_scalar else result

    support_window = (t0 - T, t0 + T)
    return f, fhat, support_window


def complex_gaussian_pair(t0=0.0, sigma=1.0, omega_c=2.0):
    r"""
    Complex Gaussian (modulated real Gaussian).

    f(t) = exp(i\omega_c*t) * exp(-(t-t0)^2/(2\sigma^2))
    f_hat(\omega) = \sigma\sqrt{2\pi} \exp(i(\omega+\omega_c)t0) \exp(-\sigma^2(\omega+\omega_c)^2/2)
    """

    def f(t):
        return np.exp(1j * omega_c * t) * np.exp(-((t - t0) ** 2) / (2 * sigma**2))

    def fhat(omega):
        return (
            sigma
            * np.sqrt(2 * np.pi)
            * np.exp(1j * (omega + omega_c) * t0)
            * np.exp(-(sigma**2) * (omega + omega_c) ** 2 / 2)
        )

    support_window = (t0 - 15 * sigma, t0 + 15 * sigma)
    return f, fhat, support_window


# Parameterized function pairs
@pytest.fixture(
    params=[
        pytest.param(gaussian_pair(), id="gaussian_centered"),
        pytest.param(gaussian_pair(t0=2.0, sigma=0.5), id="gaussian_offset_narrow"),
        pytest.param(gaussian_pair(t0=-1.0, sigma=1.5), id="gaussian_offset_wide"),
        pytest.param(ricker_pair(f0=1.0), id="ricker_default"),
        pytest.param(ricker_pair(f0=10.0), id="ricker_f0_10"),
        pytest.param(raised_cosine_pair(T=1.0, t0=0.0), id="raised_cosine_default"),
    ]
)
def real_f_pair(request):
    r"""Fixture providing various analytical function pairs. Returns (f, fhat, support_window), where
    fhat(\omega) = \int f(t)e^{i\omega t}\dt
    Here f is assumed to be real valued
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(gaussian_pair(), id="gaussian_centered"),
        pytest.param(gaussian_pair(t0=2.0, sigma=0.5), id="gaussian_offset_narrow"),
        pytest.param(gaussian_pair(t0=-1.0, sigma=1.5), id="gaussian_offset_wide"),
        pytest.param(ricker_pair(f0=1.0), id="ricker_default"),
        pytest.param(ricker_pair(f0=10.0), id="ricker_f0_10"),
        pytest.param(raised_cosine_pair(T=1.0, t0=0.0), id="raised_cosine_default"),
        pytest.param(
            complex_gaussian_pair(t0=0.0, sigma=1.0, omega_c=2.0),
            id="complex_gauss_centered",
        ),
        pytest.param(
            complex_gaussian_pair(t0=1.5, sigma=0.8, omega_c=4.0),
            id="complex_gauss_offset",
        ),
    ]
)
def f_pair(request):
    r"""Fixture providing various analytical function pairs. Returns (f, fhat, support_window), where
    fhat(\omega) = \int f(t)e^{i\omega t}\dt
    Here f may be complex valued
    """
    return request.param


# Parameterized FT parameters for basic tests (shape, spacing)
@pytest.fixture(
    params=[
        pytest.param(
            {"N": 128, "dt": 0.01, "extension_factor": 4, "t0": 0.0},
            id="N128_dt0.01_ext4",
        ),
        pytest.param(
            {"N": 256, "dt": 0.01, "extension_factor": 8, "t0": 0.0},
            id="N256_dt0.01_ext8",
        ),
        pytest.param(
            {"N": 128, "dt": 0.05, "extension_factor": 2, "t0": 0.0},
            id="N128_dt0.05_ext2",
        ),
        pytest.param(
            {"N": 64, "dt": 0.02, "extension_factor": 4, "t0": 0.0},
            id="N64_dt0.02_ext4",
        ),
    ]
)
def simulation_params(request):
    """Fixture providing various FT parameter configurations for basic tests."""
    return request.param


# Fine resolution parameters for function recovery tests
@pytest.fixture(
    params=[
        pytest.param(
            {
                "N": 256,
                "dt": 0.01,
                "extension_factor": 10,
                "refinement_factor": 8,
                "t0": 0.0,
            },
            id="fine_resolution",
        ),
        pytest.param(
            {
                "N": 128,
                "dt": 0.02,
                "extension_factor": 10,
                "refinement_factor": 4,
                "t0": 0.0,
            },
            id="medium_resolution",
        ),
    ]
)
def simulation_params_fine(request):
    """Fixture providing FT parameters with fine resolution for accurate function recovery."""
    return request.param


def test_cont_ifft_output_shape(simulation_params):
    """
    Test that the signal shape is correct
    """

    def fhat(omega):
        if type(omega) is np.ndarray:
            return np.zeros(len(omega))
            # vectorized
        return 0

    # function doesn't matter, we just need to check the shape
    _, signal = cont_ifft(fhat, simulation_params)
    requested_length = simulation_params["N"]
    desired_shape = (requested_length,)
    actual_shape = signal.shape

    print(desired_shape, actual_shape)

    # remove hanging 1's from shape
    while len(desired_shape) > 1 and desired_shape[-1] == 1:
        desired_shape = desired_shape[:-1]
    while len(actual_shape) > 1 and actual_shape[-1] == 1:
        actual_shape = actual_shape[:-1]

    assert desired_shape == actual_shape


def test_cont_ifft_time_spacing(simulation_params):
    """Test that the time spacing is correct"""

    def fhat(_):
        return np.zeros((1,))

    time, _ = cont_ifft(fhat, simulation_params)
    dt = simulation_params["dt"]
    expected_time = np.arange(simulation_params["N"]) * dt
    np.testing.assert_allclose(time, expected_time, rtol=1e-10)


def test_cont_ifft_singleton(f_pair, simulation_params_fine):
    """Test that cont_irfft recovers the original time-domain signal."""
    f, fhat, support_window = f_pair
    # Compute via inverse transform
    simulation_params_fine.update({"support_window": support_window})
    _, result = cont_ifft(fhat, simulation_params_fine)

    try:
        expected = f(
            simulation_params_fine.get("t0", 0.0)
            + np.arange(simulation_params_fine["N"]) * simulation_params_fine["dt"]
        )
    except Exception:
        # nonvectorized
        expected = np.array(
            [
                f(t)
                for t in (
                    simulation_params_fine.get("t0", 0.0)
                    + np.arange(simulation_params_fine["N"])
                    * simulation_params_fine["dt"]
                )
            ]
        )

    # uncomment this section to plot for visual inspection upon failure
    if not np.allclose(result, expected, rtol=1e-10):
        # print("cont_ifft test failed, generating plot for visual inspection.")
        # plt.figure()
        # plt.plot(time, expected.real, label="expected real", linestyle="--")
        # plt.plot(time, expected.imag, label="expected imag", linestyle="--")
        # plt.plot(time, result.real, label="result real", linestyle=":")
        # plt.plot(time, result.imag, label="result imag", linestyle=":")
        # plt.legend()
        # plt.title("cont_ifft test failure visualization")
        # plt.savefig("cont_ifft_test_failure.png")
        ...

    expected_peak = np.max(np.abs(expected))
    atol = 1e-6 * expected_peak if expected_peak != 0 else 1e-10
    np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-10)


# ---------------------------------------------------------------------------
# Axis-aware cont_ifft tests
# ---------------------------------------------------------------------------

_AXIS_PARAMS = {
    "N": 128,
    "dt": 0.02,
    "extension_factor": 6,
    "refinement_factor": 4,
    "t0": 0.0,
    "support_window": (-30.0, 30.0),
}

# A small batch of distinct spectra to transform together.
_AXIS_PAIRS = [
    gaussian_pair(t0=0.0, sigma=1.0),
    gaussian_pair(t0=2.0, sigma=0.5),
    complex_gaussian_pair(t0=0.0, sigma=1.0, omega_c=2.0),
]


def test_cont_ifft_axis_matches_axis0_batched():
    """A batched vectorized callback with frequency on axis 1 must match
    stacking independent axis=0 transforms of each spectrum."""
    fhats = [pair[1] for pair in _AXIS_PAIRS]

    def batched_fhat(omega_array):
        # shape (n_signals, n_omega) -- frequency on axis 1
        return np.stack([fh(omega_array) for fh in fhats], axis=0)

    _, batched = cont_ifft(batched_fhat, dict(_AXIS_PARAMS), axis=1)

    singles = [cont_ifft(fh, dict(_AXIS_PARAMS), axis=0)[1] for fh in fhats]
    expected = np.stack(singles, axis=1)  # (N, n_signals) -> freq(time) on axis 0

    # batched has time on axis 1: shape (n_signals, N)
    np.testing.assert_allclose(batched, expected.T, rtol=1e-10, atol=1e-12)


def test_cont_ifft_axis_with_trailing_tensor_dims():
    """Frequency on a middle axis, with trailing tensor dimensions, mirrors the
    wavefield use case (n_points, n_omega, 2, 2)."""
    fhats = [pair[1] for pair in _AXIS_PAIRS]

    def batched_fhat(omega_array):
        # (n_signals, n_omega) -> broadcast into a (n_signals, n_omega, 2, 2) tensor
        base = np.stack([fh(omega_array) for fh in fhats], axis=0)
        tensor = np.zeros((*base.shape, 2, 2), dtype=complex)
        tensor[..., 0, 0] = base
        tensor[..., 1, 1] = 2.0 * base
        return tensor

    _, batched = cont_ifft(batched_fhat, dict(_AXIS_PARAMS), axis=1)
    assert batched.shape == (len(fhats), _AXIS_PARAMS["N"], 2, 2)

    singles = [cont_ifft(fh, dict(_AXIS_PARAMS), axis=0)[1] for fh in fhats]
    expected00 = np.stack(singles, axis=0)  # (n_signals, N)
    np.testing.assert_allclose(batched[..., 0, 0], expected00, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        batched[..., 1, 1], 2.0 * expected00, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(batched[..., 0, 1], 0.0, atol=1e-12)


def test_cont_ifft_axis_scalar_fallback():
    """A scalar-only callback (forces the non-vectorized branch) respects axis."""
    fhats = [pair[1] for pair in _AXIS_PAIRS]

    def scalar_fhat(omega):
        if isinstance(omega, np.ndarray):
            msg = "scalar only"
            raise TypeError(msg)
        return np.array([fh(omega) for fh in fhats])  # shape (n_signals,)

    _, axis1 = cont_ifft(scalar_fhat, dict(_AXIS_PARAMS), axis=1)
    _, axis0 = cont_ifft(scalar_fhat, dict(_AXIS_PARAMS), axis=0)

    # axis0: (N, n_signals); axis1: (n_signals, N)
    np.testing.assert_allclose(axis1, np.moveaxis(axis0, 0, 1), rtol=1e-10, atol=1e-12)


def test_cont_ifft_axis_shape_mismatch_raises():
    """A vectorized callback whose declared axis has the wrong length errors."""

    def bad_fhat(omega_array):
        return np.zeros((3, len(omega_array) + 5), dtype=complex)

    with pytest.raises(ValueError, match="along axis"):
        cont_ifft(bad_fhat, dict(_AXIS_PARAMS), axis=1)


def test_downsample_signal_axis():
    """downsample_signal interpolates along the requested axis."""
    expanded_time = np.linspace(0.0, 1.0, 21)
    downsampled_time = np.linspace(0.0, 1.0, 6)
    # signal linear in time on axis 1: value == 2*t, independent per row/col
    signal = np.empty((3, 21, 4))
    signal[:] = (2.0 * expanded_time)[None, :, None]

    out = downsample_signal(expanded_time, downsampled_time, signal, axis=1)
    assert out.shape == (3, 6, 4)
    np.testing.assert_allclose(
        out, (2.0 * downsampled_time)[None, :, None] * np.ones((3, 1, 4)), rtol=1e-12
    )
