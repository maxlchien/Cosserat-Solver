"""
Test the Fortran dispersion 3D module.
"""

from __future__ import annotations

import numpy as np

import cosserat_solver.dim3._dispersion_core_3d_wrapper as dispersion_core_3d_wrapper
import cosserat_solver.dim3.dispersion_3d as dispersion_core_3d


def test_compare_cn_squared(material_parameters):
    """Test that the squared velocities c1^2, c2^2, c3^2, c4^2 match the Python implementation."""

    rho = material_parameters["rho"]
    lam = material_parameters["lam"]
    mu = material_parameters["mu"]
    nu = material_parameters["nu"]
    J = material_parameters["J"]
    lam_c = material_parameters["lam_c"]
    mu_c = material_parameters["mu_c"]
    nu_c = material_parameters["nu_c"]

    assert np.isclose(
        dispersion_core_3d.c1_squared(rho, lam, mu),
        dispersion_core_3d_wrapper.c1_squared(rho, lam, mu),
        atol=1e-12,
    ), (
        f"c1^2 does not match: Fortran={dispersion_core_3d_wrapper.c1_squared(rho, lam, mu)}, Python={dispersion_core_3d.c1_squared(rho, lam, mu)}"
    )
    assert np.isclose(
        dispersion_core_3d.c2_squared(rho, mu, nu),
        dispersion_core_3d_wrapper.c2_squared(rho, mu, nu),
        atol=1e-12,
    ), (
        f"c2^2 does not match: Fortran={dispersion_core_3d_wrapper.c2_squared(rho, mu, nu)}, Python={dispersion_core_3d.c2_squared(rho, mu, nu)}"
    )
    assert np.isclose(
        dispersion_core_3d.c3_squared(J, lam_c, mu_c),
        dispersion_core_3d_wrapper.c3_squared(J, lam_c, mu_c),
        atol=1e-12,
    ), (
        f"c3^2 does not match: Fortran={dispersion_core_3d_wrapper.c3_squared(J, lam_c, mu_c)}, Python={dispersion_core_3d.c3_squared(J, lam_c, mu_c)}"
    )
    assert np.isclose(
        dispersion_core_3d.c4_squared(J, mu_c, nu_c),
        dispersion_core_3d_wrapper.c4_squared(J, mu_c, nu_c),
        atol=1e-12,
    ), (
        f"c4^2 does not match: Fortran={dispersion_core_3d_wrapper.c4_squared(J, mu_c, nu_c)}, Python={dispersion_core_3d.c4_squared(J, mu_c, nu_c)}"
    )

    fortran_c_squared = dispersion_core_3d_wrapper.all_c_squared(
        rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_c_squared = dispersion_core_3d.all_c_squared(
        rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.allclose(fortran_c_squared, python_c_squared, atol=1e-12), (
        f"Squared velocities do not match:\nFortran={fortran_c_squared},\nPython={python_c_squared}"
    )

    # dict version

    fortran_c_squared_from_dict = dispersion_core_3d_wrapper.all_c_squared_from_dict(
        material_parameters
    )
    python_c_squared_from_dict = dispersion_core_3d.all_c_squared_from_dict(
        material_parameters
    )

    assert np.allclose(
        fortran_c_squared_from_dict, python_c_squared_from_dict, atol=1e-12
    ), (
        f"Squared velocities from dict do not match:\nFortran={fortran_c_squared_from_dict},\nPython={python_c_squared_from_dict}"
    )


def test_compare_cutoff_frequency(material_parameters):
    """Test that the cutoff frequency matches the Python implementation."""

    nu = material_parameters["nu"]
    J = material_parameters["J"]

    fortran_cutoff_frequency = dispersion_core_3d_wrapper.w0_squared(nu, J)
    python_cutoff_frequency = dispersion_core_3d.w0_squared(nu, J)

    assert np.isclose(fortran_cutoff_frequency, python_cutoff_frequency, atol=1e-12), (
        f"Cutoff frequency does not match: Fortran={fortran_cutoff_frequency}, Python={python_cutoff_frequency}"
    )


def test_compare_dispersion_r(material_parameters, omega_value):
    """Test that the dispersion r matches the Python implementation."""

    rho = material_parameters["rho"]
    lam = material_parameters["lam"]
    mu = material_parameters["mu"]
    nu = material_parameters["nu"]
    J = material_parameters["J"]
    lam_c = material_parameters["lam_c"]
    mu_c = material_parameters["mu_c"]
    nu_c = material_parameters["nu_c"]

    fortran_dispersion_r = dispersion_core_3d_wrapper.dispersion_r(
        omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_dispersion_r = dispersion_core_3d.dispersion_r(
        omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.isclose(fortran_dispersion_r, python_dispersion_r, atol=1e-12), (
        f"Dispersion r does not match: Fortran={fortran_dispersion_r}, Python={python_dispersion_r}"
    )


def test_compare_dispersion_s(material_parameters, omega_value):
    """Test that the dispersion s matches the Python implementation."""

    rho = material_parameters["rho"]
    lam = material_parameters["lam"]
    mu = material_parameters["mu"]
    nu = material_parameters["nu"]
    J = material_parameters["J"]
    lam_c = material_parameters["lam_c"]
    mu_c = material_parameters["mu_c"]
    nu_c = material_parameters["nu_c"]

    fortran_dispersion_s = dispersion_core_3d_wrapper.dispersion_s(
        omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_dispersion_s = dispersion_core_3d.dispersion_s(
        omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.isclose(fortran_dispersion_s, python_dispersion_s, atol=1e-12), (
        f"Dispersion s does not match: Fortran={fortran_dispersion_s}, Python={python_dispersion_s}"
    )


def test_compare_all_k(material_parameters, omega_value):
    """Test that the k values match the Python implementation."""

    rho = material_parameters["rho"]
    lam = material_parameters["lam"]
    mu = material_parameters["mu"]
    nu = material_parameters["nu"]
    J = material_parameters["J"]
    lam_c = material_parameters["lam_c"]
    mu_c = material_parameters["mu_c"]
    nu_c = material_parameters["nu_c"]

    fortran_k_values = dispersion_core_3d_wrapper.all_k(
        omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_k_values = dispersion_core_3d.all_k(
        omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.allclose(fortran_k_values, python_k_values, atol=1e-12), (
        f"k values do not match:\nFortran={fortran_k_values},\nPython={python_k_values}"
    )

    fortran_k_squared_values = dispersion_core_3d_wrapper.all_k_squared(
        omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    python_k_squared_values = dispersion_core_3d.all_k_squared(
        omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )

    assert np.allclose(fortran_k_squared_values, python_k_squared_values, atol=1e-12), (
        f"k squared values do not match:\nFortran={fortran_k_squared_values},\nPython={python_k_squared_values}"
    )

    assert np.isclose(
        dispersion_core_3d_wrapper.k1_squared(omega_value, rho, lam, mu),
        dispersion_core_3d.k1_squared(omega_value, rho, lam, mu),
        atol=1e-12,
    ), (
        f"k1^2 does not match: Fortran={dispersion_core_3d_wrapper.k1_squared(omega_value, rho, lam, mu)}, Python={dispersion_core_3d.k1_squared(omega_value, rho, lam, mu)}"
    )
    assert np.isclose(
        dispersion_core_3d_wrapper.k2_squared(
            omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        ),
        dispersion_core_3d.k2_squared(
            omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        ),
        atol=1e-12,
    ), (
        f"k2^2 does not match: Fortran={dispersion_core_3d_wrapper.k2_squared(omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)}, Python={dispersion_core_3d.k2_squared(omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)}"
    )
    assert np.isclose(
        dispersion_core_3d_wrapper.k3_squared(omega_value, nu, J, lam_c, mu_c),
        dispersion_core_3d.k3_squared(omega_value, nu, J, lam_c, mu_c),
        atol=1e-12,
    ), (
        f"k3^2 does not match: Fortran={dispersion_core_3d_wrapper.k3_squared(omega_value, nu, J, lam_c, mu_c)}, Python={dispersion_core_3d.k3_squared(omega_value, nu, J, lam_c, mu_c)}"
    )
    assert np.isclose(
        dispersion_core_3d_wrapper.k4_squared(
            omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        ),
        dispersion_core_3d.k4_squared(
            omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        ),
        atol=1e-12,
    ), (
        f"k4^2 does not match: Fortran={dispersion_core_3d_wrapper.k4_squared(omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)}, Python={dispersion_core_3d.k4_squared(omega_value, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)}"
    )
