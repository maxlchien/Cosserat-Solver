from __future__ import annotations

import numpy as np

from cosserat_solver.dim3 import dispersion


def test_k_branches(material_parameters: dict, omega_value: float) -> None:
    """
    Check that the k values all are returned either with positive imaginary part, or sign equal to that of omega when real valued.
    """
    k_values = dispersion.all_k(
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

    for k in k_values:
        if np.isclose(k.imag, 0.0, atol=1e-6):
            assert np.sign(k.real) == np.sign(omega_value), (
                f"Real k value {k} does not have the same sign as omega {omega_value}"
            )
        else:
            assert k.imag > 0, f"Imaginary part of k value {k} is not positive"


def test_k24_roots(material_parameters: dict, omega_value: float) -> None:
    """
    Test that k_2^2, k_4^2 are roots of k^2 - 2rk + s = 0.
    """
    # increase precision since there is cancellation
    material_parameters = {k: np.longdouble(v) for k, v in material_parameters.items()}
    omega_value = np.longdouble(omega_value)

    _, k2_sq, _, k4_sq = dispersion.all_k_squared(
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
    r = dispersion.dispersion_r(
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
    s = dispersion.dispersion_s(
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

    # need to use a relatively loose tolerance since there is cancellation for k2
    assert np.isclose(k2_sq**2 - 2 * r * k2_sq + s, 0, atol=1e-4), (
        f"k2^2={k2_sq} is not a root of the dispersion equation. Left hand side is {k2_sq**2 - 2 * r * k2_sq + s}"
    )
    assert np.isclose(k4_sq**2 - 2 * r * k4_sq + s, 0, atol=1e-5), (
        f"k4^2={k4_sq} is not a root of the dispersion equation. Left hand side is {k4_sq**2 - 2 * r * k4_sq + s}"
    )
