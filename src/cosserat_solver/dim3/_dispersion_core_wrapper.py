from __future__ import annotations

try:
    from cosserat_solver.dim3 import dispersion_core

    HAS_FORTRAN = True
except ImportError:
    HAS_FORTRAN = False


def c1_squared(rho: float, lam: float, mu: float) -> float:
    """Calculate the squared velocity c1^2 for the 3D Cosserat medium."""
    return dispersion_core.c1_squared(rho, lam, mu)


def c2_squared(rho: float, mu: float, nu: float) -> float:
    """Calculate the squared velocity c2^2 for the 3D Cosserat medium."""
    return dispersion_core.c2_squared(rho, mu, nu)


def c3_squared(J: float, lam_c: float, mu_c: float) -> float:
    """Calculate the squared velocity c3^2 for the 3D Cosserat medium."""
    return dispersion_core.c3_squared(J, lam_c, mu_c)


def c4_squared(J: float, mu_c: float, nu_c: float) -> float:
    """Calculate the squared velocity c4^2 for the 3D Cosserat medium."""
    return dispersion_core.c4_squared(J, mu_c, nu_c)


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
    """Calculate the squared velocity w0^2 for the 3D Cosserat medium."""
    return dispersion_core.w0_squared(nu, J)


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
    return dispersion_core.dispersion_r(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)


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
    """Calculate the s coefficient in the dispersion relation for k_2, k_4.

    This is computed by
    s = (omega^2 / c_2^2) * (omega^2 / c_4^2 - omega_0^2 / c_4^2)

    as according to Eringen (1999) equation (5.11.20).
    """
    return dispersion_core.dispersion_s(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)


def k1_squared(omega: float, rho: float, lam: float, mu: float) -> complex:
    """Calculate the squared wavenumber k1^2 for the 3D Cosserat medium."""
    result_real, result_imag = dispersion_core.k1_squared(omega, rho, lam, mu)
    return complex(result_real, result_imag)


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
) -> complex:
    """Calculate the squared wavenumber k2^2 for the 3D Cosserat medium."""
    result_real, result_imag = dispersion_core.k2_squared(
        omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    return complex(result_real, result_imag)


def k3_squared(omega: float, nu: float, J: float, lam_c: float, mu_c: float) -> complex:
    """Calculate the squared wavenumber k3^2 for the 3D Cosserat medium."""
    result_real, result_imag = dispersion_core.k3_squared(omega, nu, J, lam_c, mu_c)
    return complex(result_real, result_imag)


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
) -> complex:
    """Calculate the squared wavenumber k4^2 for the 3D Cosserat medium."""
    result_real, result_imag = dispersion_core.k4_squared(
        omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    return complex(result_real, result_imag)


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
) -> tuple[complex, complex, complex, complex]:
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
    k1_real, k1_imag = dispersion_core.k1(omega, rho, lam, mu)
    k2_real, k2_imag = dispersion_core.k2(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    k3_real, k3_imag = dispersion_core.k3(omega, nu, J, lam_c, mu_c)
    k4_real, k4_imag = dispersion_core.k4(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    return (
        complex(k1_real, k1_imag),
        complex(k2_real, k2_imag),
        complex(k3_real, k3_imag),
        complex(k4_real, k4_imag),
    )
