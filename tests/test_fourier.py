from __future__ import annotations

import numpy as np
import pytest

from cosserat_solver.fourier import cont_ifft


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


def exponential_decay_pair(alpha=1.0):
    r"""
    Exponential decay for t > 0.

    f(t) = exp(-\alpha*t) for t ≥ 0, 0 otherwise (causal signal)

    f_hat(\omega) = 1/(\alpha - i\omega)
    """

    def f(t):
        return np.exp(-alpha * t) * (t >= 0)

    def fhat(omega):
        return 1 / (alpha - 1j * omega)

    support_window = (-10 / alpha, 10 / alpha)
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
        pytest.param(exponential_decay_pair(alpha=1.0), id="exp_decay_slow"),
        pytest.param(exponential_decay_pair(alpha=3.0), id="exp_decay_fast"),
        pytest.param(ricker_pair(f0=1.0), id="ricker_default"),
        pytest.param(ricker_pair(f0=10.0), id="ricker_f0_10"),
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
        pytest.param(exponential_decay_pair(alpha=1.0), id="exp_decay_slow"),
        pytest.param(exponential_decay_pair(alpha=3.0), id="exp_decay_fast"),
        pytest.param(
            complex_gaussian_pair(t0=0.0, sigma=1.0, omega_c=2.0),
            id="complex_gauss_centered",
        ),
        pytest.param(
            complex_gaussian_pair(t0=1.5, sigma=0.8, omega_c=4.0),
            id="complex_gauss_offset",
        ),
        pytest.param(ricker_pair(f0=1.0), id="ricker_default"),
        pytest.param(ricker_pair(f0=10.0), id="ricker_f0_10"),
    ]
)
def f_pair(request):
    r"""Fixture providing various analytical function pairs. Returns (f, fhat, support_window), where
    fhat(\omega) = \int f(t)e^{i\omega t}\dt
    Here f may be complex valued
    """
    return request.param


# Parameterized FT parameters
@pytest.fixture(
    params=[
        pytest.param(
            {"N": 128, "dt": 0.01, "oversample_rate": 4, "t0": 0.0},
            id="N128_dt0.01_os4",
        ),
        pytest.param(
            {"N": 256, "dt": 0.01, "oversample_rate": 8, "t0": 0.0},
            id="N256_dt0.01_os8",
        ),
        pytest.param(
            {"N": 128, "dt": 0.05, "oversample_rate": 4, "t0": 0.0},
            id="N128_dt0.05_os4",
        ),
        pytest.param(
            {"N": 64, "dt": 0.02, "oversample_rate": 4, "t0": 0.0}, id="N64_dt0.02_os4"
        ),
    ]
)
def ft_params(request):
    """Fixture providing various FT parameter configurations."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param({"N": 128, "dt": 0.1, "oversample_rate": 2}, id="os2"),
        pytest.param({"N": 128, "dt": 0.1, "oversample_rate": 4}, id="os4"),
        pytest.param({"N": 64, "dt": 0.1, "oversample_rate": 8}, id="os8"),
    ]
)
def ft_params_oversampled(request):
    """Fixture providing oversampled FT parameters."""
    return request.param


def test_cont_ifft_output_shape(ft_params):
    """
    Test that the signal shape is correct
    """

    def fhat(omega):
        if type(omega) is np.ndarray:
            return np.zeros(len(omega))
            # vectorized
        return 0

    # function doesn't matter, we just need to check the shape
    _, signal = cont_ifft(fhat, ft_params)
    requested_length = ft_params["N"]
    desired_shape = (requested_length,)
    actual_shape = signal.shape

    print(desired_shape, actual_shape)

    # remove hanging 1's from shape
    while len(desired_shape) > 1 and desired_shape[-1] == 1:
        desired_shape = desired_shape[:-1]
    while len(actual_shape) > 1 and actual_shape[-1] == 1:
        actual_shape = actual_shape[:-1]

    assert desired_shape == actual_shape


def test_cont_ifft_time_spacing(ft_params):
    """Test that the time spacing is correct"""

    def fhat(_):
        return np.zeros((1,))

    time, _ = cont_ifft(fhat, ft_params)
    dt = ft_params["dt"]
    expected_time = np.arange(ft_params["N"]) * dt
    np.testing.assert_allclose(time, expected_time, rtol=1e-10)


def test_cont_ifft_singleton(f_pair, ft_params):
    """Test that cont_irfft recovers the original time-domain signal."""
    f, fhat, support_window = f_pair
    # Compute via inverse transform
    ft_params.update({"support_window": support_window})
    _, result = cont_ifft(fhat, ft_params)

    try:
        expected = f(
            ft_params.get("t0", 0.0) + np.arange(ft_params["N"]) * ft_params["dt"]
        )
    except Exception:
        # nonvectorized
        expected = np.array(
            [
                f(t)
                for t in (
                    ft_params.get("t0", 0.0)
                    + np.arange(ft_params["N"]) * ft_params["dt"]
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
