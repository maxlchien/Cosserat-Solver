"""
Test the fortran wrapper module _dispersion_core_wrapper.
Due to roudning issues in the callback function, only the following aspects are tested:
- That coefficients A, B, C, and c_pm, c_pm_prime are the same as their Python equivalents, up to double float tolerance
- That dispersion(r) is zero, up to double float tolerance

The complete test suite is not run here but instead in the python equivalent test, since all computations can be performed
via mpmath there, without casting through to double in the C wrapper.

In particular, we do NOT test that dispersion(r, c_pm(r)) == 0, as the cast through C in c_pm (when called from pytest as opposed to Fortran)
leads to issues with precision.
"""

from __future__ import annotations

import numpy as np

from cosserat_solver._dispersion_core_wrapper import DispersionHelperFortran
from cosserat_solver.dispersion import DispersionHelper as DispersionHelperPython


def test_compare_dispersion_A(material_parameters, wave_parameters):
    """Test that dispersion coefficient A matches the Python implementation."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]

    dispersion_helper_fortran = DispersionHelperFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    dispersion_helper_python = DispersionHelperPython(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    A_fortran = dispersion_helper_fortran._dispersion_A(k)
    A_fortran = complex(A_fortran)
    A_python = dispersion_helper_python._dispersion_A(k)
    A_python = complex(A_python)

    assert np.isclose(A_fortran.real, A_python.real, atol=1e-12), (
        f"Real parts of dispersion_A do not match: Fortran={A_fortran.real}, Python={A_python.real}"
    )
    assert np.isclose(A_fortran.imag, A_python.imag, atol=1e-12), (
        f"Imaginary parts of dispersion_A do not match: Fortran={A_fortran.imag}, Python={A_python.imag}"
    )


def test_compare_dispersion_B(material_parameters, wave_parameters):
    """Test that dispersion coefficient B matches the Python implementation."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]

    dispersion_helper_fortran = DispersionHelperFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    dispersion_helper_python = DispersionHelperPython(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    B_fortran = dispersion_helper_fortran._dispersion_B(k)
    B_fortran = complex(B_fortran)
    B_python = dispersion_helper_python._dispersion_B(k)
    B_python = complex(B_python)

    assert np.isclose(B_fortran.real, B_python.real, atol=1e-12), (
        f"Real parts of dispersion_B do not match: Fortran={B_fortran.real}, Python={B_python.real}"
    )
    assert np.isclose(B_fortran.imag, B_python.imag, atol=1e-12), (
        f"Imaginary parts of dispersion_B do not match: Fortran={B_fortran.imag}, Python={B_python.imag}"
    )


def test_compare_dispersion_C(material_parameters, wave_parameters):
    """Test that dispersion coefficient C matches the Python implementation."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]

    dispersion_helper_fortran = DispersionHelperFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    dispersion_helper_python = DispersionHelperPython(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    C_fortran = dispersion_helper_fortran._dispersion_C(k)
    C_fortran = complex(C_fortran)
    C_python = dispersion_helper_python._dispersion_C(k)
    C_python = complex(C_python)

    assert np.isclose(C_fortran.real, C_python.real, atol=1e-12), (
        f"Real parts of dispersion_C do not match: Fortran={C_fortran.real}, Python={C_python.real}"
    )
    assert np.isclose(C_fortran.imag, C_python.imag, atol=1e-12), (
        f"Imaginary parts of dispersion_C do not match: Fortran={C_fortran.imag}, Python={C_python.imag}"
    )


def test_compare_c_pm(material_parameters, wave_parameters, branch):
    """Test that c_pm matches the Python implementation."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]

    dispersion_helper_fortran = DispersionHelperFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    dispersion_helper_python = DispersionHelperPython(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    c_fortran = dispersion_helper_fortran.c_pm(k, branch)
    c_fortran = complex(c_fortran)
    c_python = dispersion_helper_python.c_pm(k, branch)
    c_python = complex(c_python)

    assert np.isclose(c_fortran.real, c_python.real, atol=1e-12), (
        f"Real parts of c_pm do not match: Fortran={c_fortran.real}, Python={c_python.real}"
    )
    assert np.isclose(c_fortran.imag, c_python.imag, atol=1e-12), (
        f"Imaginary parts of c_pm do not match: Fortran={c_fortran.imag}, Python={c_python.imag}"
    )


def test_compare_cpm_prime(material_parameters, wave_parameters, branch):
    """Test that c_pm_prime matches the Python implementation."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]

    dispersion_helper_fortran = DispersionHelperFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    dispersion_helper_python = DispersionHelperPython(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    cpm_prime_fortran = dispersion_helper_fortran.c_pm_prime(k, branch)
    cpm_prime_fortran = complex(cpm_prime_fortran)
    cpm_prime_python = dispersion_helper_python.c_pm_prime(k, branch)
    cpm_prime_python = complex(cpm_prime_python)

    assert np.isclose(cpm_prime_fortran.real, cpm_prime_python.real, atol=1e-12), (
        f"Real parts of c_pm_prime do not match: Fortran={cpm_prime_fortran.real}, Python={cpm_prime_python.real}"
    )
    assert np.isclose(cpm_prime_fortran.imag, cpm_prime_python.imag, atol=1e-12), (
        f"Imaginary parts of c_pm_prime do not match: Fortran={cpm_prime_fortran.imag}, Python={cpm_prime_python.imag}"
    )


def test_dispersion_zero(material_parameters, wave_parameters, branch):
    """Test that the dispersion relation is satisfied (dispersion_zero() == 0)."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]

    dispersion_helper_fortran = DispersionHelperFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=30,
    )

    dispersion_value = dispersion_helper_fortran._dispersion_zero(k, branch)
    dispersion_value = complex(dispersion_value)

    assert np.isclose(dispersion_value.real, 0.0, atol=1e-12), (
        f"Real part of dispersion relation not close to zero: {dispersion_value.real}"
    )
    assert np.isclose(dispersion_value.imag, 0.0, atol=1e-12), (
        f"Imaginary part of dispersion relation not close to zero: {dispersion_value.imag}"
    )
