"""
Test the Fortran integrator_core wrapper module (_integrator_core_wrapper).

Following the pattern from dispersion tests, we avoid precision loss by:
- Comparing intermediate Fortran results with Python equivalents up to double precision
- Verifying final outputs (integral results) match Python

We do NOT test round-trip precision where intermediate values are cast back to Python and forth,
as this introduces precision loss through the C FFI.
"""

from __future__ import annotations

import numpy as np

from cosserat_solver._integrator_core_wrapper import IntegratorFortran
from cosserat_solver.integrator import Integrator


def test_denom_fortran_vs_python(material_parameters, k_value, omega_value, branch):
    """Test that Fortran denom matches Python implementation up to double precision."""
    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = k_value
    omega = omega_value

    integrator_fortran = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    integrator_python = Integrator(
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

    denom_fortran = integrator_fortran.denom(k, omega, branch)
    denom_fortran = complex(denom_fortran)

    denom_python = integrator_python.denom(k, omega, branch)
    denom_python = complex(denom_python)

    assert np.isclose(denom_fortran.real, denom_python.real, atol=1e-12), (
        f"Real parts of denom do not match: Fortran={denom_fortran.real}, Python={denom_python.real}"
    )
    assert np.isclose(denom_fortran.imag, denom_python.imag, atol=1e-12), (
        f"Imaginary parts of denom do not match: Fortran={denom_fortran.imag}, Python={denom_python.imag}"
    )


def test_denom_prime_fortran_vs_python(
    material_parameters, k_value, omega_value, branch
):
    """Test that Fortran denom_prime matches Python implementation up to double precision."""
    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = k_value
    omega = omega_value

    integrator_fortran = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    integrator_python = Integrator(
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

    denom_prime_fortran = integrator_fortran.denom_prime(k, omega, branch)
    denom_prime_fortran = complex(denom_prime_fortran)

    denom_prime_python = integrator_python.denom_prime(k, omega, branch)
    denom_prime_python = complex(denom_prime_python)

    assert np.isclose(denom_prime_fortran.real, denom_prime_python.real, atol=1e-12), (
        f"Real parts of denom_prime do not match: Fortran={denom_prime_fortran.real}, Python={denom_prime_python.real}"
    )
    assert np.isclose(denom_prime_fortran.imag, denom_prime_python.imag, atol=1e-12), (
        f"Imaginary parts of denom_prime do not match: Fortran={denom_prime_fortran.imag}, Python={denom_prime_python.imag}"
    )


def test_get_r2_poles_and_branches_fortran_vs_python(material_parameters, omega_value):
    """Test that Fortran pole computation matches Python implementation."""
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

    integrator_fortran = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    integrator_python = Integrator(
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

    r2_poles_fortran, _ = integrator_fortran.get_r2_poles_and_branches(omega)
    r2_poles_python = integrator_python._get_r2_poles_and_branches(omega)

    assert len(r2_poles_fortran) == len(r2_poles_python), (
        f"Number of poles mismatch: Fortran={len(r2_poles_fortran)}, Python={len(r2_poles_python)}"
    )

    for i, (r2_f, r2_p_data) in enumerate(
        zip(r2_poles_fortran, r2_poles_python, strict=False)
    ):
        r2_p = r2_p_data[0]  # Extract r2 value from (r2, branch) tuple
        assert np.isclose(r2_f.real, complex(r2_p).real, atol=1e-10), (
            f"Real parts of r2 pole {i} do not match: Fortran={r2_f.real}, Python={complex(r2_p).real}"
        )
        assert np.isclose(r2_f.imag, complex(r2_p).imag, atol=1e-10), (
            f"Imaginary parts of r2 pole {i} do not match: Fortran={r2_f.imag}, Python={complex(r2_p).imag}"
        )


def test_integral_3_0_fortran_vs_python(
    material_parameters, omega_value, norm_x_value, branch
):
    """Test that Fortran integral_3_0 matches Python implementation."""
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
    normx = norm_x_value

    integrator_fortran = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    integrator_python = Integrator(
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

    integral_fortran = integrator_fortran.integral_3_0(omega, normx, branch)
    integral_fortran = complex(integral_fortran)

    integral_python = integrator_python.integral_3_0(normx, omega, branch)
    integral_python = complex(integral_python)

    # Use relative + absolute tolerance to handle both small and large integral values.
    # FFI precision loss (double precision only) affects pole computation, which propagates to integrals.
    # Relative tolerance of 1e-10 handles large values; absolute tolerance of 1e-8 handles small values.
    assert np.isclose(
        integral_fortran.real, integral_python.real, rtol=1e-10, atol=1e-8
    ), (
        f"Real parts of integral_3_0 do not match: Fortran={integral_fortran.real}, Python={integral_python.real}"
    )
    assert np.isclose(
        integral_fortran.imag, integral_python.imag, rtol=1e-10, atol=1e-8
    ), (
        f"Imaginary parts of integral_3_0 do not match: Fortran={integral_fortran.imag}, Python={integral_python.imag}"
    )


def test_integral_3_2_fortran_vs_python(
    material_parameters, omega_value, norm_x_value, branch
):
    """Test that Fortran integral_3_2 matches Python implementation."""
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
    normx = norm_x_value

    integrator_fortran = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    integrator_python = Integrator(
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

    integral_fortran = integrator_fortran.integral_3_2(omega, normx, branch)
    integral_fortran = complex(integral_fortran)

    integral_python = integrator_python.integral_3_2(normx, omega, branch)
    integral_python = complex(integral_python)

    # Use relative + absolute tolerance to handle both small and large integral values.
    # FFI precision loss (double precision only) affects pole computation, which propagates to integrals.
    # Relative tolerance of 1e-10 handles large values; absolute tolerance of 1e-8 handles small values.
    assert np.isclose(
        integral_fortran.real, integral_python.real, rtol=1e-10, atol=1e-8
    ), (
        f"Real parts of integral_3_2 do not match: Fortran={integral_fortran.real}, Python={integral_python.real}"
    )
    assert np.isclose(
        integral_fortran.imag, integral_python.imag, rtol=1e-10, atol=1e-8
    ), (
        f"Imaginary parts of integral_3_2 do not match: Fortran={integral_fortran.imag}, Python={integral_python.imag}"
    )


def test_integral_2_1_fortran_vs_python(
    material_parameters, omega_value, norm_x_value, branch
):
    """Test that Fortran integral_2_1 matches Python implementation."""
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
    normx = norm_x_value

    integrator_fortran = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    integrator_python = Integrator(
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

    integral_fortran = integrator_fortran.integral_2_1(omega, normx, branch)
    integral_fortran = complex(integral_fortran)

    integral_python = integrator_python.integral_2_1(normx, omega, branch)
    integral_python = complex(integral_python)

    # Use relative + absolute tolerance to handle both small and large integral values.
    # FFI precision loss (double precision only) affects pole computation, which propagates to integrals.
    # Relative tolerance of 1e-10 handles large values; absolute tolerance of 1e-8 handles small values.
    assert np.isclose(
        integral_fortran.real, integral_python.real, rtol=1e-10, atol=1e-8
    ), (
        f"Real parts of integral_2_1 do not match: Fortran={integral_fortran.real}, Python={integral_python.real}"
    )
    assert np.isclose(
        integral_fortran.imag, integral_python.imag, rtol=1e-10, atol=1e-8
    ), (
        f"Imaginary parts of integral_2_1 do not match: Fortran={integral_fortran.imag}, Python={integral_python.imag}"
    )


def test_integral_1_0_fortran_vs_python(
    material_parameters, omega_value, norm_x_value, branch
):
    """Test that Fortran integral_1_0 matches Python implementation."""
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
    normx = norm_x_value

    integrator_fortran = IntegratorFortran(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
    )

    integrator_python = Integrator(
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

    integral_fortran = integrator_fortran.integral_1_0(omega, normx, branch)
    integral_fortran = complex(integral_fortran)

    integral_python = integrator_python.integral_1_0(normx, omega, branch)
    integral_python = complex(integral_python)

    # Use relative + absolute tolerance to handle both small and large integral values.
    # FFI precision loss (double precision only) affects pole computation, which propagates to integrals.
    # Relative tolerance of 1e-10 handles large values; absolute tolerance of 1e-8 handles small values.
    assert np.isclose(
        integral_fortran.real, integral_python.real, rtol=1e-10, atol=1e-8
    ), (
        f"Real parts of integral_1_0 do not match: Fortran={integral_fortran.real}, Python={integral_python.real}"
    )
    assert np.isclose(
        integral_fortran.imag, integral_python.imag, rtol=1e-10, atol=1e-8
    ), (
        f"Imaginary parts of integral_1_0 do not match: Fortran={integral_fortran.imag}, Python={integral_python.imag}"
    )
