from __future__ import annotations

import numpy as np
import pytest

from cosserat_solver.ricker import Ricker


def ricker_time_domain(t, f0):
    """Ricker wavelet in time domain."""
    a = (np.pi * f0) ** 2
    return (1 - 2 * a * t**2) * np.exp(-(a * t**2))


@pytest.mark.parametrize("f0", [10.0, 25.0, 50.0])
def test_ricker_spectrum(f0):
    """
    Check that the ricker spectrum fhat satisfies
    fhat = int f(t)exp(+i omega t)dt
    """
    # pick our omega of interest
    omegas = np.linspace(0.1 * 2 * np.pi * f0, 3 * 2 * np.pi * f0, 50)
    result = np.array([Ricker({"f0": f0}).spectrum(omega) for omega in omegas])

    def numerically_integrate(omega):
        # Time domain setup
        dt = 1 / (20 * f0)
        t = np.arange(-2 / f0, 2 / f0, dt)
        signal = ricker_time_domain(t, f0)
        integrand = signal * np.exp(1j * omega * t)
        return np.trapezoid(integrand, t)

    expected = np.array([numerically_integrate(omega) for omega in omegas])

    np.testing.assert_allclose(result, expected, rtol=1e-10)
