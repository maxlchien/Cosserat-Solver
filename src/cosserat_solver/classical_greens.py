from __future__ import annotations

import numpy as np


def _validate_x_3d(x: np.ndarray) -> tuple[float, np.ndarray]:
    if len(x) != 3:
        err = f"Spatial location x must have length 3 for 3D problems. Got length {len(x)}."
        raise ValueError(err)

    r = float(np.linalg.norm(x))
    if r == 0.0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)

    r_hat = np.asarray(x, dtype=float) / r
    return r, r_hat


def _kelvin_static_green(
    r: float, r_hat: np.ndarray, lam: float, mu: float
) -> np.ndarray:
    """Static 3D Kelvin tensor for isotropic elasticity."""
    # Equivalent to 1/(16*pi*mu*(1-nu_p)*r) * ((3-4*nu_p) I + r_hat r_hat^T)
    # but expressed directly in terms of Lamé parameters.
    common = 1.0 / (8.0 * np.pi * mu * (lam + 2.0 * mu) * r)
    i_coeff = lam + 3.0 * mu
    rr_coeff = lam + mu
    return common * (i_coeff * np.identity(3) + rr_coeff * np.outer(r_hat, r_hat))


def _hessian_eikr_over_r(k: complex, r: float, r_hat: np.ndarray) -> np.ndarray:
    """Hessian of phi(r)=exp(i k r)/r for radial 3D fields."""
    exp_term = np.exp(1j * k * r)
    phi = exp_term / r
    phi_prime_over_r = (1j * k * r - 1.0) * exp_term / (r**3)
    radial_coeff = -(k**2) * phi - 3.0 * phi_prime_over_r
    return phi_prime_over_r * np.identity(3) + radial_coeff * np.outer(r_hat, r_hat)


def _classical_displacement_green_3d(
    x: np.ndarray,
    omega: float,
    rho: float,
    lam: float,
    mu: float,
) -> np.ndarray:
    """
    Dynamic displacement Green tensor in frequency domain for isotropic 3D elasticity.

    Form used:
      G = (1/(4*pi*mu)) * [ phi_s I + (1/k_s^2) (H_s - H_p) ]
    where phi_a = exp(i k_a r)/r and H_a = grad grad phi_a.

    A static Kelvin fallback is used near omega=0 for numerical stability.
    """
    r, r_hat = _validate_x_3d(x)

    # Static fallback around zero frequency avoids k_s^-2 cancellation at omega~0.
    if abs(omega) < 1e-12:
        return _kelvin_static_green(r, r_hat, lam, mu)

    cp = np.sqrt((lam + 2.0 * mu) / rho)
    cs = np.sqrt(mu / rho)

    kp = omega / cp
    ks = omega / cs

    phi_s = np.exp(1j * ks * r) / r
    h_s = _hessian_eikr_over_r(ks, r, r_hat)
    h_p = _hessian_eikr_over_r(kp, r, r_hat)

    return (1.0 / (4.0 * np.pi * mu)) * (phi_s * np.identity(3) + (h_s - h_p) / (ks**2))


def greens_displacement_force(
    x: np.ndarray,
    omega: float,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> np.ndarray:
    """
    Classical-elastic displacement response block with Cosserat-compatible signature.

    Unused Cosserat parameters are accepted for API compatibility.
    """
    _ = (nu, J, lam_c, mu_c, nu_c)

    g = np.zeros((6, 3), dtype=np.complex128)
    g[:3, :] = _classical_displacement_green_3d(x, omega, rho, lam, mu)
    return g


def greens_rotation_force(
    x: np.ndarray,
    omega: float,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> np.ndarray:
    """Classical elasticity has no independent microrotation force response."""
    _ = (x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    return np.zeros((6, 3), dtype=np.complex128)


def greens_mixed_force(
    x: np.ndarray,
    omega: float,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
) -> np.ndarray:
    """6x6 mixed-force Green tensor, matching the existing 3D Cosserat API."""
    g = np.zeros((6, 6), dtype=np.complex128)
    g[:, :3] = greens_displacement_force(
        x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    g[:, 3:] = greens_rotation_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    return g


def greens_mixed_force_from_dict(
    x: np.ndarray,
    omega: float,
    material_params: dict,
) -> np.ndarray:
    return greens_mixed_force(
        x,
        omega,
        material_params["rho"],
        material_params["lam"],
        material_params["mu"],
        material_params["nu"],
        material_params["J"],
        material_params["lam_c"],
        material_params["mu_c"],
        material_params["nu_c"],
    )
