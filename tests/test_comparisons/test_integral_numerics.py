"""
Numerical verification of integral computations.

Compares residue theorem results (from both Python and Fortran implementations)
against direct numerical integration using SciPy's quad with LowLevelCallable.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from mpmath import mp
from scipy.integrate import IntegrationWarning, quad

import cosserat_solver.consts as consts
from cosserat_solver._integrator_core_wrapper import IntegratorFortran
from cosserat_solver.integrator import Integrator

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

mp.dps = consts.TEST_PRECISION

INTEGRAND_IDS = {
    "3_0": 0,
    "3_2": 1,
    "2_1": 2,
    "1_0": 3,
}


def _quad_lowlevel_integrand(
    integrator: IntegratorFortran,
    integrand_id: int,
    omega: complex,
    norm_x: complex,
    branch: int,
    contour_shift: float = 1e-8,
) -> tuple[complex, float, bool]:
    """
    Numerically integrate the integrand along the real axis with a small imaginary shift.

    Parameters
    ----------
    integrator : IntegratorFortran
        The Fortran integrator with LowLevelCallable support
    integrand_id : int
        The integrand ID (0 for 3_0, 1 for 3_2, 2 for 2_1, 3 for 1_0)
    omega : complex
        The angular frequency
    norm_x : complex
        The norm of the spatial coordinate
    branch : int
        The branch selection (+1 or -1)
    contour_shift : float
        The imaginary shift for the integration contour

    Returns
    -------
    result : complex
        The numerical integral result
    error : float
        The estimated error in the integral
    """
    # Get LowLevelCallable for real component (component=0)
    llc_real, _ = integrator.lowlevel_integrand(
        integrand_id, omega, norm_x, branch, component=0, contour_shift=contour_shift
    )

    # Get LowLevelCallable for imaginary component (component=1)
    llc_imag, _ = integrator.lowlevel_integrand(
        integrand_id, omega, norm_x, branch, component=1, contour_shift=contour_shift
    )

    # Integrate from -infinity to +infinity
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", IntegrationWarning)
        real_result, real_error = quad(llc_real, -np.inf, np.inf)
        imag_result, imag_error = quad(llc_imag, -np.inf, np.inf)

    result = complex(real_result, imag_result)
    error = np.sqrt(real_error**2 + imag_error**2)

    had_warning = any(
        issubclass(warning.category, IntegrationWarning) for warning in caught_warnings
    )

    return result, error, had_warning


def _mpmath_integral(
    integrator: Integrator,
    integrand_id: int,
    omega: mp.mpf | mp.mpc,
    norm_x: mp.mpf | mp.mpc,
    branch: int,
    rho: mp.mpf,
    mu: mp.mpf,
    nu: mp.mpf,
    J: mp.mpf,
) -> complex:
    old_dps = mp.dps
    mp.dps = 50

    def integrand(r):
        r = r + mp.mpc(0, 1e-16)  # contour shift
        c_pm = integrator.dispersion_helper.c_pm(r, branch)
        denom = (
            (mu + nu) * r**2
            - rho * omega**2
            + 2 * nu * mp.mpc(0, 1) * mp.sqrt(rho / J) * c_pm
        )
        if integrand_id == INTEGRAND_IDS["3_0"]:
            hankel = mp.hankel1(0, r * norm_x)
            numer = r**3 / (r**2 + abs(c_pm) ** 2)
        elif integrand_id == INTEGRAND_IDS["3_2"]:
            hankel = mp.hankel1(2, r * norm_x)
            numer = r**3 / (r**2 + abs(c_pm) ** 2)
        elif integrand_id == INTEGRAND_IDS["2_1"]:
            hankel = mp.hankel1(1, r * norm_x)
            numer = (r**2 * c_pm) / (r**2 + abs(c_pm) ** 2)
        elif integrand_id == INTEGRAND_IDS["1_0"]:
            hankel = mp.hankel1(0, r * norm_x)
            numer = (r * abs(c_pm) ** 2) / (r**2 + abs(c_pm) ** 2)
        else:
            return mp.mpc(0, 0)

        return (1 / denom) * numer * hankel

    result = mp.quad(integrand, [-mp.inf, mp.inf]) / (8 * mp.pi)

    if integrand_id == INTEGRAND_IDS["2_1"]:
        result = result * mp.sqrt(rho / J)
    elif integrand_id == INTEGRAND_IDS["1_0"]:
        result = result * (rho / J)

    mp.dps = old_dps
    return complex(result)


def test_integration_3_0(material_parameters, omega_value, norm_x_value, branch):
    r"""Test that the residue integration for I_{3,0} is consistent with numerical calculation

    I_{3,0,\pm}=\frac{1}{8\pi}\int_{-\infty}^\infty \frac{1}{(\mu+\nu)r^2-\rho\omega^2+2\nu i \sqrt{\frac{\rho}{j}} c_\pm} \frac{r^3}{r^2+\abs{c_\pm}^2}H_0^{(1)}(r\norm{x})\dr
    """

    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = omega_value

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    python_result = integrator.integral_3_0(norm_x_value, omega, branch)

    fortran_integrator = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    fortran_result = fortran_integrator.integral_3_0(
        complex(omega), complex(norm_x_value), branch
    )

    poles = integrator.get_poles_and_branches(omega)
    d1_prime = integrator.denom_prime(poles[0][0], omega, poles[0][1])
    d2_prime = integrator.denom_prime(poles[1][0], omega, poles[1][1])

    numerical_result, _, had_warning = _quad_lowlevel_integrand(
        fortran_integrator,
        INTEGRAND_IDS["3_0"],
        complex(omega),
        complex(norm_x_value),
        branch,
        contour_shift=1e-8,
    )
    numerical_result = numerical_result / (8 * np.pi)

    # Cast mpmath result to complex for numpy comparison
    python_result_float = complex(python_result)

    # if there is an error or warning, compute with more precision
    if had_warning or not np.isclose(python_result_float, numerical_result, rtol=1e-5):
        numerical_result = _mpmath_integral(
            integrator,
            INTEGRAND_IDS["3_0"],
            omega,
            norm_x_value,
            branch,
            rho,
            mu,
            nu,
            J,
        )

    assert np.isclose(python_result_float, fortran_result, rtol=1e-5), (
        f"Integral I_3,0 mismatch (python vs fortran) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )
    assert np.isclose(python_result_float, numerical_result, rtol=1e-5), (
        f"Integral I_3,0 mismatch (python vs numerical) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )
    assert np.isclose(fortran_result, numerical_result, rtol=1e-5), (
        f"Integral I_3,0 mismatch (fortran vs numerical) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )


def test_integration_3_2(material_parameters, omega_value, norm_x_value, branch):
    r"""Test that the residue integration for I_{3,2} is consistent with numerical calculation

    I_{3,2,\pm}=\frac{1}{8\pi}\int_{-\infty}^\infty \frac{1}{(\mu+\nu)r^2-\rho\omega^2+2\nu i \sqrt{\frac{\rho}{j}} c_\pm} \frac{r^3}{r^2+\abs{c_\pm}^2}H_2^{(1)}(r\norm{x})\dr
    """

    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = omega_value

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    python_result = integrator.integral_3_2(norm_x_value, omega, branch)

    fortran_integrator = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    fortran_result = fortran_integrator.integral_3_2(
        complex(omega), complex(norm_x_value), branch
    )

    poles = integrator.get_poles_and_branches(omega)
    d1_prime = integrator.denom_prime(poles[0][0], omega, poles[0][1])
    d2_prime = integrator.denom_prime(poles[1][0], omega, poles[1][1])

    numerical_result, _, had_warning = _quad_lowlevel_integrand(
        fortran_integrator,
        INTEGRAND_IDS["3_2"],
        complex(omega),
        complex(norm_x_value),
        branch,
        contour_shift=1e-8,
    )
    numerical_result = numerical_result / (8 * np.pi)

    # Cast mpmath result to complex for numpy comparison
    python_result_float = complex(python_result)

    # if there is an error or warning, compute with more precision
    if had_warning or not np.isclose(python_result_float, numerical_result, rtol=1e-5):
        numerical_result = _mpmath_integral(
            integrator,
            INTEGRAND_IDS["3_2"],
            omega,
            norm_x_value,
            branch,
            rho,
            mu,
            nu,
            J,
        )

    assert np.isclose(python_result_float, fortran_result, rtol=1e-5), (
        f"Integral I_3,2 mismatch (python vs fortran) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )
    assert np.isclose(python_result_float, numerical_result, rtol=1e-5), (
        f"Integral I_3,2 mismatch (python vs numerical) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )
    assert np.isclose(fortran_result, numerical_result, rtol=1e-5), (
        f"Integral I_3,2 mismatch (fortran vs numerical) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )


def test_integration_2_1(material_parameters, omega_value, norm_x_value, branch):
    r"""Test that the residue integration for I_{2,1} is consistent with numerical calculation

    I_{2,1,\pm}=\frac{1}{8\pi} \sqrt{\frac{\rho}{j}} \int_{-\infty}^\infty \frac{1}{(\mu+\nu)r^2-\rho\omega^2+2\nu i \sqrt{\frac{\rho}{j}} c_\pm} \frac{r^2 c_\pm}{r^2+\abs{c_\pm}^2}H_1^{(1)}(r\norm{x})\dr
    """

    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = omega_value

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    python_result = integrator.integral_2_1(norm_x_value, omega, branch)

    fortran_integrator = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    fortran_result = fortran_integrator.integral_2_1(
        complex(omega), complex(norm_x_value), branch
    )

    poles = integrator.get_poles_and_branches(omega)
    d1_prime = integrator.denom_prime(poles[0][0], omega, poles[0][1])
    d2_prime = integrator.denom_prime(poles[1][0], omega, poles[1][1])

    numerical_result, _, had_warning = _quad_lowlevel_integrand(
        fortran_integrator,
        INTEGRAND_IDS["2_1"],
        complex(omega),
        complex(norm_x_value),
        branch,
        contour_shift=1e-8,
    )
    numerical_result = numerical_result / (8 * np.pi) * np.sqrt(float(rho / J))

    # Cast mpmath result to complex for numpy comparison
    python_result_float = complex(python_result)

    # if there is an error or warning, compute with more precision
    if had_warning or not np.isclose(python_result_float, numerical_result, rtol=1e-5):
        numerical_result = _mpmath_integral(
            integrator,
            INTEGRAND_IDS["2_1"],
            omega,
            norm_x_value,
            branch,
            rho,
            mu,
            nu,
            J,
        )

    assert np.isclose(python_result_float, fortran_result, rtol=1e-5), (
        f"Integral I_2,1 mismatch (python vs fortran) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )
    assert np.isclose(python_result_float, numerical_result, rtol=1e-5), (
        f"Integral I_2,1 mismatch (python vs numerical) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )
    assert np.isclose(fortran_result, numerical_result, rtol=1e-5), (
        f"Integral I_2,1 mismatch (fortran vs numerical) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )


def test_integration_1_0(material_parameters, omega_value, norm_x_value, branch):
    r"""Test that the residue integration for I_{1,0} is consistent with numerical calculation

    I_{1,0,\pm}=\frac{1}{8\pi} \frac{\rho}{j}\int_{-\infty}^\infty \frac{1}{(\mu+\nu)r^2-\rho\omega^2+2\nu i \sqrt{\frac{\rho}{j}} c_\pm} \frac{r \abs{c_\pm}^2}{r^2+\abs{c_\pm}^2}H_0^{(1)}(r\norm{x})\dr
    """

    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = omega_value

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    python_result = integrator.integral_1_0(norm_x_value, omega, branch)

    fortran_integrator = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    fortran_result = fortran_integrator.integral_1_0(
        complex(omega), complex(norm_x_value), branch
    )

    poles = integrator.get_poles_and_branches(omega)
    d1_prime = integrator.denom_prime(poles[0][0], omega, poles[0][1])
    d2_prime = integrator.denom_prime(poles[1][0], omega, poles[1][1])

    numerical_result, _, had_warning = _quad_lowlevel_integrand(
        fortran_integrator,
        INTEGRAND_IDS["1_0"],
        complex(omega),
        complex(norm_x_value),
        branch,
        contour_shift=1e-8,
    )
    numerical_result = numerical_result / (8 * np.pi) * float(rho / J)

    # Cast mpmath result to complex for numpy comparison
    python_result_float = complex(python_result)

    # if there is an error or warning, compute with more precision
    if had_warning or not np.isclose(python_result_float, numerical_result, rtol=1e-5):
        numerical_result = _mpmath_integral(
            integrator,
            INTEGRAND_IDS["1_0"],
            omega,
            norm_x_value,
            branch,
            rho,
            mu,
            nu,
            J,
        )

    assert np.isclose(python_result_float, fortran_result, rtol=1e-5), (
        f"Integral I_1,0 mismatch (python vs fortran) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )
    assert np.isclose(python_result_float, numerical_result, rtol=1e-5), (
        f"Integral I_1,0 mismatch (python vs numerical) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )
    assert np.isclose(fortran_result, numerical_result, rtol=1e-5), (
        f"Integral I_1,0 mismatch (fortran vs numerical) for params: {params}, omega: {omega}, norm_x: {norm_x_value}, branch: {branch} \
            d_prime (first pole): {d1_prime}, d_prime (second pole): {d2_prime}"
    )
