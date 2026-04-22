from __future__ import annotations

import numpy as np

# define the constants


def c1_squared(rho: float, lam: float, mu: float) -> float:
    """Calculate the squared velocity c1^2 for the 3D Cosserat medium."""
    return (lam + 2 * mu) / rho


def c2_squared(rho: float, mu: float, nu: float) -> float:
    """Calculate the squared velocity c2^2 for the 3D Cosserat medium."""
    return (mu + nu) / rho


def c3_squared(J: float, lam_c: float, mu_c: float) -> float:
    """Calculate the squared velocity c3^2 for the 3D Cosserat medium."""
    return (lam_c + 2 * mu_c) / J


def c4_squared(J: float, mu_c: float, nu_c: float) -> float:
    """Calculate the squared velocity c4^2 for the 3D Cosserat medium."""
    return (mu_c + nu_c) / J


def all_c_squared(
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> tuple[float, float, float, float]:
    """Calculate all squared velocities c1^2, c2^2, c3^2, c4^2 for the 3D Cosserat medium."""
    return (
        c1_squared(rho, lam, mu),
        c2_squared(rho, mu, nu),
        c3_squared(J, lam_c, mu_c),
        c4_squared(J, mu_c, nu_c),
    )


def all_c_squared_from_dict(material_params: dict) -> tuple[float, float, float, float]:
    """Calculate all squared velocities c1^2, c2^2, c3^2, c4^2 for the 3D Cosserat medium from a material parameters dictionary."""
    return all_c_squared(
        material_params["rho"],
        material_params["lam"],
        material_params["mu"],
        material_params["nu"],
        material_params["J"],
        material_params["lam_c"],
        material_params["mu_c"],
        material_params["nu_c"],
    )


def w0_squared(nu: float, J: float) -> float:
    """Calculate the squared cutoff frequency w0^2 for the 3D Cosserat medium."""
    return 4 * nu / J


def dispersion_r(
    omega: float,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> float:
    """Calculate the r coefficient in the dispersion relation for k_2, k_4.

    This is computed by
    r = (1 + c_2^2 / c_4^2) * (omega^2 / c_2^2) * (1/2) - (1 - J * w_0^2 / (4 * rho * c_2^2)) * w_0^2 / (2 * c_4^2)

    as according to Eringen (1999) equation (5.11.20).
    """
    _, c2_sq, _, c4_sq = all_c_squared(rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    w0_sq = w0_squared(nu, J)
    term1 = (1 + c2_sq / c4_sq) * (omega**2 / c2_sq) * (1 / 2)
    term2 = (1 - J * w0_sq / (4 * rho * c2_sq)) * w0_sq / (2 * c4_sq)
    return term1 - term2


def dispersion_s(
    omega: float,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> float:
    """Calculate the s coefficient in the dispersion relation for k_1, k_3.

    This is computed by
    s = (omega^2 / c_2^2) * (omega^2 / c_4^2 - omega_0^2 / c_4^2)

    as according to Eringen (1999) equation (5.11.20).
    """
    _, c2_sq, _, c4_sq = all_c_squared(rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    w0_sq = w0_squared(nu, J)
    return (omega**2 / c2_sq) * (omega**2 / c4_sq - w0_sq / c4_sq)


def k1_squared(omega: float, rho: float, lam: float, mu: float) -> float:
    """Calculate the squared wavenumber k1^2 for the 3D Cosserat medium."""
    return omega**2 / c1_squared(rho, lam, mu)


def k2_squared(
    omega: float,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> float:
    """Calculate the squared wavenumber k2^2 for the 3D Cosserat medium."""
    r = dispersion_r(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    s = dispersion_s(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    return r + np.sqrt(r**2 - s)


def k3_squared(
    omega: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
) -> float:
    """Calculate the squared wavenumber k3^2 for the 3D Cosserat medium."""
    return (omega**2 - w0_squared(nu, J)) / c3_squared(J, lam_c, mu_c)


def k4_squared(
    omega: float,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> float:
    """Calculate the squared wavenumber k4^2 for the 3D Cosserat medium."""
    r = dispersion_r(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    s = dispersion_s(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    return r - np.sqrt(r**2 - s)


def all_k_squared(
    omega: float,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> tuple[float, float, float, float]:
    """Calculate all squared wavenumbers k1^2, k2^2, k3^2, k4^2 for the 3D Cosserat medium."""
    return (
        k1_squared(omega, rho, lam, mu),
        k2_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c),
        k3_squared(omega, nu, J, lam_c, mu_c),
        k4_squared(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c),
    )


def all_k(
    omega: float,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> tuple[complex, complex, complex, complex]:
    """Calculate all wavenumbers k1, k2, k3, k4 for the 3D Cosserat medium."""
    k1_sq, k2_sq, k3_sq, k4_sq = all_k_squared(
        omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    k1 = np.emath.sqrt(k1_sq)
    k2 = np.emath.sqrt(k2_sq)
    k3 = np.emath.sqrt(k3_sq)
    k4 = np.emath.sqrt(k4_sq)
    sign = np.sign(omega)
    k1 = sign * k1 if k1_sq >= 0 else k1
    k2 = sign * k2 if k2_sq >= 0 else k2
    k3 = sign * k3 if k3_sq >= 0 else k3
    k4 = sign * k4 if k4_sq >= 0 else k4
    # TODO: address this properly
    return (
        k1,
        k2,
        k3,
        k4,
    )
