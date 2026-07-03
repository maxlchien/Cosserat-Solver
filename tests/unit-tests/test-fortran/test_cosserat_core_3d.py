"""
Test the Fortran Elastic 3D wrapper module (_elastic_core_3d_wrapper).

Verifies that the Green's functions are all the same as their Python equivalents, up to double float tolerance.
"""

from __future__ import annotations

import numpy as np

import cosserat_solver._cosserat_core_3d_wrapper as _cosserat_core_3d_wrapper
import cosserat_solver.cosserat_3d as cosserat_3d

np.set_printoptions(precision=2)


def test_compare_greens_mixed(material_parameters, omega_value, location_3d):
    """Test that the mixed Green's function matches the Python implementation."""

    # first test from dict

    fortran_greens_from_dict = _cosserat_core_3d_wrapper.greens_mixed_force_from_dict(
        location_3d, omega_value, material_parameters
    )
    python_greens_from_dict = cosserat_3d.greens_mixed_force_from_dict(
        location_3d, omega_value, material_parameters
    )

    assert np.allclose(fortran_greens_from_dict, python_greens_from_dict, atol=1e-12), (
        f"Mixed Green's function from dict does not match:\nFortran={fortran_greens_from_dict},\nPython={python_greens_from_dict}"
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

    fortran_greens = _cosserat_core_3d_wrapper.greens_mixed_force(
        location_3d, omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_greens = cosserat_3d.greens_mixed_force(
        location_3d, omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.allclose(fortran_greens, python_greens, atol=1e-12), (
        f"Mixed Green's function does not match:\nFortran={fortran_greens},\nPython={python_greens}"
    )


def test_compare_greens_displacement(material_parameters, omega_value, location_3d):
    """Test that the displacement Green's function matches the Python implementation."""

    # first test from dict

    fortran_greens_from_dict = (
        _cosserat_core_3d_wrapper.greens_displacement_force_from_dict(
            location_3d, omega_value, material_parameters
        )
    )
    python_greens_from_dict = cosserat_3d.greens_displacement_force_from_dict(
        location_3d, omega_value, material_parameters
    )

    assert np.allclose(fortran_greens_from_dict, python_greens_from_dict, atol=1e-12), (
        f"Displacement Green's function from dict does not match:\nFortran={fortran_greens_from_dict},\nPython={python_greens_from_dict}"
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

    fortran_greens = _cosserat_core_3d_wrapper.greens_displacement_force(
        location_3d, omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_greens = cosserat_3d.greens_displacement_force(
        location_3d, omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.allclose(fortran_greens, python_greens, atol=1e-12), (
        f"Displacement Green's function does not match:\nFortran={fortran_greens},\nPython={python_greens}"
    )


def test_compare_greens_rotation(material_parameters, omega_value, location_3d):
    """Test that the rotation Green's function matches the Python implementation."""

    # first test from dict

    fortran_greens_from_dict = (
        _cosserat_core_3d_wrapper.greens_rotation_force_from_dict(
            location_3d, omega_value, material_parameters
        )
    )
    python_greens_from_dict = cosserat_3d.greens_rotation_force_from_dict(
        location_3d, omega_value, material_parameters
    )

    assert np.allclose(fortran_greens_from_dict, python_greens_from_dict, atol=1e-12), (
        f"Rotation Green's function from dict does not match:\nFortran={fortran_greens_from_dict},\nPython={python_greens_from_dict}"
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

    fortran_greens = _cosserat_core_3d_wrapper.greens_rotation_force(
        location_3d, omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_greens = cosserat_3d.greens_rotation_force(
        location_3d, omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.allclose(fortran_greens, python_greens, atol=1e-12), (
        f"Rotation Green's function does not match:\nFortran={fortran_greens},\nPython={python_greens}"
    )


def test_compare_greens_displacement_static(material_parameters, location_3d):
    """Test that the static displacement Green's function matches the Python implementation."""

    # There are currently no dict-reading functions for the static Green's functions
    rho = material_parameters["rho"]
    lam = material_parameters["lam"]
    mu = material_parameters["mu"]
    nu = material_parameters["nu"]
    J = material_parameters["J"]
    lam_c = material_parameters["lam_c"]
    mu_c = material_parameters["mu_c"]
    nu_c = material_parameters["nu_c"]

    fortran_greens = _cosserat_core_3d_wrapper.greens_displacement_force_static(
        location_3d, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_greens = cosserat_3d.greens_displacement_force_static(
        location_3d, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.allclose(fortran_greens, python_greens, atol=1e-12), (
        f"Static displacement Green's function does not match:\nFortran={fortran_greens},\nPython={python_greens}"
    )


def test_compare_greens_rotation_static(material_parameters, location_3d):
    """Test that the static rotation Green's function matches the Python implementation."""

    # There are currently no dict-reading functions for the static Green's functions
    rho = material_parameters["rho"]
    lam = material_parameters["lam"]
    mu = material_parameters["mu"]
    nu = material_parameters["nu"]
    J = material_parameters["J"]
    lam_c = material_parameters["lam_c"]
    mu_c = material_parameters["mu_c"]
    nu_c = material_parameters["nu_c"]

    fortran_greens = _cosserat_core_3d_wrapper.greens_rotation_force_static(
        location_3d, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_greens = cosserat_3d.greens_rotation_force_static(
        location_3d, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.allclose(fortran_greens, python_greens, atol=1e-12), (
        f"Static rotation Green's function does not match:\nFortran={fortran_greens},\nPython={python_greens}"
    )
