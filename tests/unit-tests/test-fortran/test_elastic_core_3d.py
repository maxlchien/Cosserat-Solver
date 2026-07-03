"""
Test the Fortran Elastic 3D wrapper module (_elastic_core_3d_wrapper).

Verifies that the Green's functions are all the same as their Python equivalents, up to double float tolerance.
"""

from __future__ import annotations

import numpy as np

import cosserat_solver._elastic_core_3d_wrapper as _elastic_core_3d_wrapper
import cosserat_solver.elastic_3d as elastic_3d


def test_compare_greens_mixed(material_parameters, omega_value, location_3d):
    """Test that the mixed Green's function matches the Python implementation."""

    # first test from dict

    fortran_greens_from_dict = _elastic_core_3d_wrapper.greens_mixed_force_from_dict(
        location_3d, omega_value, material_parameters
    )
    python_greens_from_dict = elastic_3d.greens_mixed_force_from_dict(
        location_3d, omega_value, material_parameters
    )

    assert np.allclose(fortran_greens_from_dict, python_greens_from_dict, atol=1e-12), (
        f"Mixed Green's function from dict does not match: Fortran={fortran_greens_from_dict}, Python={python_greens_from_dict}"
    )

    # then test from individual parameters

    rho = material_parameters["rho"]
    lam = material_parameters["lam"]
    mu = material_parameters["mu"]
    nu = material_parameters["nu"]
    J = material_parameters["J"]
    lam_c = material_parameters["lam_c"]
    mu_c = material_parameters["mu_c"]
    nu_c = material_parameters["nu_c"]

    fortran_greens = _elastic_core_3d_wrapper.greens_mixed_force(
        location_3d, omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_greens = elastic_3d.greens_mixed_force(
        location_3d, omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.allclose(fortran_greens, python_greens, atol=1e-12), (
        f"Mixed Green's function does not match: Fortran={fortran_greens}, Python={python_greens}"
    )
