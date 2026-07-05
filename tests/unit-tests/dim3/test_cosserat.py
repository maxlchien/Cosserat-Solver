from __future__ import annotations

import numpy as np

from cosserat_solver.dim3 import cosserat


def test_shape(material_parameters) -> None:
    """
    Test that the shape of the Greens function is correct.
    """
    x = np.array([1300.0, 700.0, 2000.0])
    omega = 2 * np.pi

    g = cosserat.greens_mixed_force(
        x,
        omega,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        material_parameters["nu"],
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )
    g_disp = cosserat.greens_displacement_force(
        x,
        omega,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        material_parameters["nu"],
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )
    g_rot = cosserat.greens_rotation_force(
        x,
        omega,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        material_parameters["nu"],
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )

    assert g.shape == (6, 6)
    # verify off blocks are zero
    assert np.allclose(g_disp[:, 3:], 0)
    assert np.allclose(g_rot[:, :3], 0)


def test_block_concatenation(
    material_parameters: dict, omega_value: float, location_3d: np.ndarray
) -> None:
    """
    Check that the mixed force is the sum of the displacement and rotation forces.
    """
    x = location_3d

    g_disp = cosserat.greens_displacement_force(
        x,
        omega_value,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        material_parameters["nu"],
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )
    g_rot = cosserat.greens_rotation_force(
        x,
        omega_value,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        material_parameters["nu"],
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )
    g_mixed = cosserat.greens_mixed_force(
        x,
        omega_value,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        material_parameters["nu"],
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )

    g_combined = np.zeros_like(g_mixed)
    g_combined[:, :3] = g_disp[:, :]
    g_combined[:, 3:] = g_rot[:, :]

    assert np.allclose(g_mixed, g_combined)


def test_hermitian_conjugacy(
    material_parameters: dict, omega_value: float, location_3d: np.ndarray
) -> None:
    """
    Test that the Greens function is Hermitian conjugate symmetric for a range of nu values.
    """
    x = location_3d

    g = cosserat.greens_mixed_force(
        x,
        omega_value,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        material_parameters["nu"],
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )

    g_minus = cosserat.greens_mixed_force(
        x,
        -omega_value,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        material_parameters["nu"],
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )

    denom = max(np.linalg.norm(g), np.linalg.norm(g_minus), 1e-10)
    rel_err = np.linalg.norm(g - g_minus.conj()) / denom
    assert rel_err < 1.0e-8, f"rel_err={rel_err}, g: {g}, g_minus: {g_minus}"
