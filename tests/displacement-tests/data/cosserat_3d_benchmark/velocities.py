from __future__ import annotations

import numpy as np
import yaml


def _safe_sqrt(value: float) -> float:
    return np.sqrt(value) if value > 0.0 else float("nan")


def _k_branches_from_omega(
    omega: float,
    rho: float,
    mu: float,
    nu: float,
    inertia_j: float,
    mu_c: float,
    nu_c: float,
    w0: float,
) -> tuple[float, float]:
    """Return (k_plus, k_minus) for the coupled transverse branches at fixed omega."""
    a = mu + nu
    b = mu_c + nu_c

    center = (
        rho * omega * omega * b
        + inertia_j * a * (omega * omega - w0 * w0)
        + inertia_j * nu * w0 * w0
    )
    discriminant = center * center - 4.0 * a * b * rho * inertia_j * omega * omega * (
        omega * omega - w0 * w0
    )

    if discriminant < 0.0:
        return float("nan"), float("nan")

    sqrt_disc = np.sqrt(discriminant)
    denom = 2.0 * a * b

    k2_plus = (center - sqrt_disc) / denom
    k2_minus = (center + sqrt_disc) / denom

    return _safe_sqrt(k2_plus), _safe_sqrt(k2_minus)


def _group_velocity_from_omega_k(
    omega: float,
    k: float,
    rho: float,
    mu: float,
    nu: float,
    inertia_j: float,
    mu_c: float,
    nu_c: float,
    w0: float,
) -> float:
    """Evaluate d(omega)/d(k) from the implicit coupled-transverse dispersion relation."""
    if not np.isfinite(k):
        return float("nan")

    a = mu + nu
    b = mu_c + nu_c

    num = (
        2.0 * k * (rho * b + inertia_j * a) * omega * omega
        - 2.0 * inertia_j * w0 * w0 * a * k
        - 4.0 * k * k * k * a * b
        + 2.0 * inertia_j * nu * w0 * w0 * k
    )
    denom = 4.0 * rho * inertia_j * omega * omega * omega + 2.0 * omega * (
        -rho * w0 * w0 * inertia_j - rho * b * k * k - inertia_j * a * k * k
    )

    return num / denom if denom != 0.0 else float("nan")


def compute_group_velocities(
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    inertia_j: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
    f0: float,
) -> dict[str, float]:
    """Compute the four group velocities at source frequency f0 (Hz).

    Modes returned are:
    - alpha (longitudinal displacement)
    - alpha_c (longitudinal microrotation)
    - beta_plus (coupled transverse + branch)
    - beta_minus (coupled transverse - branch)
    """
    omega = 2.0 * np.pi * f0
    w0 = 2.0 * np.sqrt(nu / inertia_j)

    alpha_group = np.sqrt((lam + 2.0 * mu) / rho)

    if omega <= w0:
        alpha_c_group = float("nan")
    else:
        alpha_c_group = np.sqrt(
            (lam_c + 2.0 * mu_c) / inertia_j * (1.0 - (w0 * w0) / (omega * omega))
        )

    k_plus, k_minus = _k_branches_from_omega(
        omega=omega,
        rho=rho,
        mu=mu,
        nu=nu,
        inertia_j=inertia_j,
        mu_c=mu_c,
        nu_c=nu_c,
        w0=w0,
    )

    beta_plus_group = _group_velocity_from_omega_k(
        omega=omega,
        k=k_plus,
        rho=rho,
        mu=mu,
        nu=nu,
        inertia_j=inertia_j,
        mu_c=mu_c,
        nu_c=nu_c,
        w0=w0,
    )
    beta_minus_group = _group_velocity_from_omega_k(
        omega=omega,
        k=k_minus,
        rho=rho,
        mu=mu,
        nu=nu,
        inertia_j=inertia_j,
        mu_c=mu_c,
        nu_c=nu_c,
        w0=w0,
    )

    return {
        "P wave": alpha_group,
        "Coupled P Wave": alpha_c_group,
        "Transverse Acoustic": beta_plus_group,
        "Transverse Optical": beta_minus_group,
    }


def load_group_velocities_from_params(params_yaml_path: str) -> dict[str, float]:
    """Load four wave-mode group velocities from Fourier params YAML."""
    with open(params_yaml_path, encoding="utf-8") as file_handle:
        params = yaml.safe_load(file_handle)

    material = params["material_params"]
    source = params["source_params"]

    low_freq_group_velocities = compute_group_velocities(
        rho=float(material["rho"]),
        lam=float(material["lam"]),
        mu=float(material["mu"]),
        nu=float(material["nu"]),
        inertia_j=float(material["J"]),
        lam_c=float(material["lam_c"]),
        mu_c=float(material["mu_c"]),
        nu_c=float(material["nu_c"]),
        f0=float(source["f0"]) / 5,
    )

    f0_group_velocities = compute_group_velocities(
        rho=float(material["rho"]),
        lam=float(material["lam"]),
        mu=float(material["mu"]),
        nu=float(material["nu"]),
        inertia_j=float(material["J"]),
        lam_c=float(material["lam_c"]),
        mu_c=float(material["mu_c"]),
        nu_c=float(material["nu_c"]),
        f0=float(source["f0"]),
    )

    hi_freq_group_velocities = compute_group_velocities(
        rho=float(material["rho"]),
        lam=float(material["lam"]),
        mu=float(material["mu"]),
        nu=float(material["nu"]),
        inertia_j=float(material["J"]),
        lam_c=float(material["lam_c"]),
        mu_c=float(material["mu_c"]),
        nu_c=float(material["nu_c"]),
        f0=float(source["f0"]) * 2.5,
    )
    # Most energy is contained in the spectrum [f0 / 5, 2.5f0], so the highe rgroup velocity should lead most of the energy content of the wave arrival
    return {
        "P wave": np.nanmax(
            [
                low_freq_group_velocities["P wave"],
                f0_group_velocities["P wave"],
                hi_freq_group_velocities["P wave"],
            ]
        ),
        "Coupled P Wave": np.nanmax(
            [
                low_freq_group_velocities["Coupled P Wave"],
                f0_group_velocities["Coupled P Wave"],
                hi_freq_group_velocities["Coupled P Wave"],
            ]
        ),
        "Transverse Acoustic": np.nanmax(
            [
                low_freq_group_velocities["Transverse Acoustic"],
                f0_group_velocities["Transverse Acoustic"],
                hi_freq_group_velocities["Transverse Acoustic"],
            ]
        ),
        "Transverse Optical": np.nanmax(
            [
                low_freq_group_velocities["Transverse Optical"],
                f0_group_velocities["Transverse Optical"],
                hi_freq_group_velocities["Transverse Optical"],
            ]
        ),
    }
