"""
Tests that as when the system becomes uncoupled, it approaches two classical systems.
"""

from __future__ import annotations

import numpy as np

from cosserat_solver.dim3 import cosserat, elastic


def test_uncoupled_displacement_force_matches_classical(
    material_parameters: dict, location_3d: np.ndarray
) -> None:
    """
    When nu = 0, the 3D displacement force block must reduce to classical isotropic elasticity.
    Tests only the implementation for nu = 0, and not the limit.
    """
    # When nu = 0, the 3D displacement force block must reduce
    # to classical isotropic elasticity.
    x = location_3d
    omega = 2 * np.pi

    nu = 0

    g_cosserat = cosserat.greens_displacement_force(
        x,
        omega,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        nu,
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )[:3, :]

    g_classical = elastic.greens_displacement_force(
        x,
        omega,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        nu,
        1,
        0,
        0,
        0,
    )[:3, :]

    denom = np.linalg.norm(g_classical)
    assert denom > 0.0
    rel_err = np.linalg.norm(g_cosserat - g_classical) / denom

    # The limit should be numerically exact up to roundoff.
    assert rel_err < 1.0e-8, f"nu={nu}, rel_err={rel_err}"


def test_limiting_uncoupled_displacement_force_matches_classical(
    material_parameters: dict, location_3d: np.ndarray
) -> None:
    """
    In the decoupling limit nu->0, the 3D displacement force block must reduce to classical isotropic elasticity.
    Tests the limit as nu approaches 0, and not the implementation for nu = 0.
    """
    # In the decoupling limit nu->0, the 3D displacement force block must reduce
    # to classical isotropic elasticity.
    x = location_3d
    omega = 2 * np.pi

    for ratio in (1.0e-7, 1.0e-8, 1.0e-9):
        nu = ratio * float(material_parameters["J"])

        g_cosserat = cosserat.greens_displacement_force(
            x,
            omega,
            material_parameters["rho"],
            material_parameters["lam"],
            material_parameters["mu"],
            nu,
            material_parameters["J"],
            material_parameters["lam_c"],
            material_parameters["mu_c"],
            material_parameters["nu_c"],
        )[:3, :]

        g_classical = elastic.greens_displacement_force(
            x,
            omega,
            material_parameters["rho"],
            material_parameters["lam"],
            material_parameters["mu"],
            nu,
            1,
            0,
            0,
            0,
        )[:3, :]

        denom = np.linalg.norm(g_classical)
        assert denom > 0.0
        rel_err = np.linalg.norm(g_cosserat - g_classical) / denom

        # Relaxed tolerance for the limit.
        assert rel_err < 1.0e-5, f"nu={nu}, rel_err={rel_err}"


def test_uncoupled_rotation_force_matches_classical(
    material_parameters: dict, location_3d: np.ndarray
) -> None:
    """
    When nu = 0, the 3D rotation force block must reduce to classical isotropic elasticity (with parameters adjusted accordingly).
    Tests only the implementation for nu = 0, and not the limit.
    """
    # When nu = 0, the 3D rotation force block must reduce
    # to classical isotropic elasticity (for a rotation force, with parameters adjusted accordingly)
    x = location_3d
    omega = 2 * np.pi

    nu = 0

    g_cosserat = cosserat.greens_rotation_force(
        x,
        omega,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        nu,
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )[3:, :]

    g_classical = elastic.greens_displacement_force(
        x,
        omega,
        material_parameters["J"],
        material_parameters["lam_c"] - 2 * material_parameters["nu_c"],
        material_parameters["mu_c"]
        + material_parameters[
            "nu_c"
        ],  # this conversion gets us to the classical setup for rotation
        nu,
        1,
        0,
        0,
        0,
    )[:3, :]

    denom = np.linalg.norm(g_classical)
    assert denom > 0.0
    rel_err = np.linalg.norm(g_cosserat - g_classical) / denom

    # The limit should be numerically exact up to roundoff.
    assert rel_err < 1.0e-8, (
        f"nu={nu}, rel_err={rel_err}, Classical: {g_classical}, Cosserat: {g_cosserat}"
    )


def test_limiting_uncoupled_rotation_force_matches_classical(
    material_parameters: dict, location_3d: np.ndarray
) -> None:
    """
    In the decoupling limit nu->0, the 3D rotation force block must reduce to classical isotropic elasticity (for a rotation force, with parameters adjusted accordingly).
    Tests the limit as nu approaches 0, and not the implementation for nu = 0.
    """
    # In the decoupling limit nu->0, the 3D rotation force block must reduce
    # to classical isotropic elasticity (for a rotation force, with parameters adjusted accordingly)
    x = location_3d

    omega = 2 * np.pi

    for ratio in (1.0e-7, 1.0e-8, 1.0e-9):
        nu = ratio * float(material_parameters["J"])

        g_cosserat = cosserat.greens_rotation_force(
            x,
            omega,
            material_parameters["rho"],
            material_parameters["lam"],
            material_parameters["mu"],
            nu,
            material_parameters["J"],
            material_parameters["lam_c"],
            material_parameters["mu_c"],
            material_parameters["nu_c"],
        )[3:, :]

        g_classical = elastic.greens_displacement_force(
            x,
            omega,
            material_parameters["J"],
            material_parameters["lam_c"] - 2 * material_parameters["nu_c"],
            material_parameters["mu_c"]
            + material_parameters[
                "nu_c"
            ],  # this conversion gets us to the classical setup for rotation
            nu,
            1,
            0,
            0,
            0,
        )[:3, :]

        denom = np.linalg.norm(g_classical)
        assert denom > 0.0
        rel_err = np.linalg.norm(g_cosserat - g_classical) / denom

        # Relaxed tolerance for the limit.
        assert rel_err < 1.0e-5, (
            f"nu={nu}, rel_err={rel_err}, Classical: {g_classical}, Cosserat: {g_cosserat}"
        )


def test_uncoupled_forces_are_symmetric(
    material_parameters: dict, location_3d: np.ndarray
) -> None:
    """
    When nu = 0, the 3D displacement and rotation force blocks must be symmetric.
    Tests only the implementation for nu = 0, and not the limit.
    """
    # In the decoupling limit nu->0, the 3D displacement and rotation force blocks must be symmetric.
    x = location_3d
    omega = 2 * np.pi

    nu = 0

    g_cosserat_disp = cosserat.greens_displacement_force(
        x,
        omega,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        nu,
        material_parameters["J"],
        material_parameters["lam_c"],
        material_parameters["mu_c"],
        material_parameters["nu_c"],
    )[:3, :]

    g_cosserat_rot = cosserat.greens_rotation_force(
        x,
        omega,
        material_parameters["J"],
        material_parameters["lam_c"] - 2 * material_parameters["nu_c"],
        material_parameters["mu_c"] + material_parameters["nu_c"],
        nu,
        material_parameters["rho"],
        material_parameters["lam"],
        material_parameters["mu"],
        0,
    )[3:, :]

    # The two blocks should be equal up to roundoff.
    assert np.allclose(g_cosserat_disp, g_cosserat_rot, rtol=1.0e-8)


def test_limiting_uncoupled_forces_are_symmetric(
    material_parameters: dict, location_3d: np.ndarray
) -> None:
    """
    In the decoupling limit nu->0, the 3D displacement and rotation force blocks must be symmetric.
    Tests the limit as nu approaches 0, and not the implementation for nu = 0.
    """
    # In the decoupling limit nu->0, the 3D displacement and rotation force blocks must match.
    x = location_3d
    omega = 2 * np.pi

    for ratio in (1.0e-7, 1.0e-8, 1.0e-9):
        nu = ratio * float(material_parameters["J"])

        g_cosserat_disp = cosserat.greens_displacement_force(
            x,
            omega,
            material_parameters["J"],
            material_parameters["lam_c"]
            - 2
            * material_parameters[
                "nu_c"
            ],  # J, lam_c, mu_c, nu_c are chosen to fit the limit better so we use them
            material_parameters["mu_c"] + material_parameters["nu_c"],
            nu,  # but nu is not interchangeable
            material_parameters["rho"],
            material_parameters["lam"],
            material_parameters["mu"],
            0,
        )[:3, :]

        g_cosserat_rot = cosserat.greens_rotation_force(
            x,
            omega,
            material_parameters["rho"],
            material_parameters["lam"],
            material_parameters["mu"],
            nu,  # but nu is not interchangeable
            material_parameters["J"],
            material_parameters["lam_c"],
            material_parameters["mu_c"],
            material_parameters["nu_c"],
        )[3:, :]

        g_elastic = elastic.greens_displacement_force(
            x,
            omega,
            material_parameters["J"],
            material_parameters["lam_c"] - 2 * material_parameters["nu_c"],
            material_parameters["mu_c"] + material_parameters["nu_c"],
            nu,
            1,
            0,
            0,
            0,
        )[
            :3, :
        ]  # for scaling the error, we use the elastic solution which is the common limit of both blocks

        denom = np.linalg.norm(g_elastic)
        assert denom > 0.0
        rel_err = np.linalg.norm(g_cosserat_disp - g_cosserat_rot) / denom

        # Relaxed tolerance for the limit.
        assert rel_err < 1.0e-5, (
            f"nu={nu}, rel_err={rel_err}, g_disp: {g_cosserat_disp}, g_rot: {g_cosserat_rot}, elastic: {g_elastic}"
        )
