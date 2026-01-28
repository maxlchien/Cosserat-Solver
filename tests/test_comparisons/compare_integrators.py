"""
Compare integrator implementations between Fortran and Python.

This module tests that the Fortran integrator produces the same r2 pole values
as the Python implementation. Note that we only compare the r2 values themselves,
not the denominators at those poles, because:

1. Poles are computed in Fortran with high precision (quad precision)
2. They are returned to Python as double precision through the FFI boundary
3. When passed back to Fortran functions, this precision loss causes
   denominators to not be exactly zero anymore
4. The actual integrals still match well because the residue computation
   is done entirely within Fortran at high precision
"""

from __future__ import annotations

import numpy as np

from cosserat_solver._integrator_core_wrapper import IntegratorFortran
from cosserat_solver.integrator import Integrator


def test_r2_poles_fortran_vs_python(material_parameters, omega_value):
    """Test that Fortran r2 poles match Python implementation.

    The r2 pole values computed by Fortran should match Python's high-precision
    results to within double precision accuracy.
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

    r2_poles_fortran, branches_fortran = integrator_fortran.get_r2_poles_and_branches(
        omega
    )
    r2_poles_python = integrator_python._get_r2_poles_and_branches(omega)

    assert len(r2_poles_fortran) == len(r2_poles_python), (
        f"Number of poles mismatch: Fortran={len(r2_poles_fortran)}, Python={len(r2_poles_python)}"
    )

    for i, (r2_f, branch_f, r2_p_data) in enumerate(
        zip(r2_poles_fortran, branches_fortran, r2_poles_python, strict=False)
    ):
        r2_p, branch_p = r2_p_data  # Extract r2 and branch from tuple

        # Check that r2 values match to double precision
        assert np.isclose(r2_f.real, complex(r2_p).real, rtol=1e-14), (
            f"Real parts of r2 pole {i} do not match: Fortran={r2_f.real}, Python={complex(r2_p).real}"
        )
        assert np.isclose(r2_f.imag, complex(r2_p).imag, atol=1e-10), (
            f"Imaginary parts of r2 pole {i} do not match: Fortran={r2_f.imag}, Python={complex(r2_p).imag}"
        )

        # Check that branches match
        assert branch_f == branch_p, (
            f"Branch for pole {i} does not match: Fortran={branch_f}, Python={branch_p}"
        )
