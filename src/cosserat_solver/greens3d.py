from __future__ import annotations

import numpy as np

import cosserat_solver.dispersion3d as dispersion3d


def greens_mixed_force_from_dict(
    x: np.ndarray,
    omega: float,
    material_params: dict,
) -> np.ndarray:
    """
    Compute the Green's function for the response to a mixed force source in a 3D Cosserat medium.

    Parameters:
        x: np.ndarray
            The spatial location where the Green's function is evaluated. Should be a 3D vector.
        omega: float
            The angular frequency at which to evaluate the Green's function.
        material_params: dict
            A dictionary containing the material parameters:
            - 'rho': Density
            - 'lam': Lamé's first parameter
            - 'mu': Shear modulus
            - 'nu': Cosserat couple modulus
            - 'J': Micro-inertia
            - 'lam_c': Cosserat Lamé's first parameter
            - 'mu_c': Cosserat shear modulus
            - 'nu_c': Cosserat couple modulus

    Returns:
        np.ndarray
            A 6x6 complex array representing the Green's function for response to a mixed force source.
            The indices correspond to (displacement component, rotation component).
    """

    # Extract material parameters
    rho = material_params["rho"]
    lam = material_params["lam"]
    mu = material_params["mu"]
    nu = material_params["nu"]
    J = material_params["J"]
    lam_c = material_params["lam_c"]
    mu_c = material_params["mu_c"]
    nu_c = material_params["nu_c"]

    return greens_mixed_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)


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
    """
    Compute the Green's function for the response to a mixed force source in a 3D Cosserat medium.

    Parameters:
        x: np.ndarray
            The spatial location where the Green's function is evaluated. Should be a 3D vector.
        omega: float
            The angular frequency at which to evaluate the Green's function.
        rho: float
            Density of the medium.
        lam: float
            Lamé's first parameter.
        mu: float
            Shear modulus.
        nu: float
            Cosserat couple modulus.
        J: float
            Micro-inertia.
        lam_c: float
            Cosserat Lamé's first parameter.
        mu_c: float
            Cosserat shear modulus.
        nu_c: float
            Cosserat couple modulus.

    Returns:
        np.ndarray
            A 6x6 complex array representing the Green's function for response to a mixed force source.
            The indices correspond to (displacement component, rotation component).
    """

    if len(x) != 3:
        err = f"Spatial location x must have length 3 for 3D problems. Got length {len(x)}."
        raise ValueError(err)

    G = np.zeros((6, 6), dtype=complex)
    G[:, :3] = greens_displacement_force(
        x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    G[:, 3:] = greens_rotation_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    return G


def greens_displacement_force_from_dict(
    x: np.ndarray,
    omega: float,
    material_params: dict,
) -> np.ndarray:
    """
    Compute the Green's function for the response to a displacement force source in a 3D Cosserat medium.

    Parameters:
        x: np.ndarray
            The spatial location where the Green's function is evaluated. Should be a 3D vector.
        omega: float
            The angular frequency at which to evaluate the Green's function.
        material_params: dict
            A dictionary containing the material parameters:
            - 'rho': Density
            - 'lam': Lamé's first parameter
            - 'mu': Shear modulus
            - 'nu': Cosserat couple modulus
            - 'J': Micro-inertia
            - 'lam_c': Cosserat Lamé's first parameter
            - 'mu_c': Cosserat shear modulus
            - 'nu_c': Cosserat couple modulus

    Returns:
        np.ndarray
            A 6x3 complex array representing the Green's function for response to a displacement force source.
            The indices correspond to (displacement component, rotation component).
    """
    # Extract material parameters
    rho = material_params["rho"]
    lam = material_params["lam"]
    mu = material_params["mu"]
    nu = material_params["nu"]
    J = material_params["J"]
    lam_c = material_params["lam_c"]
    mu_c = material_params["mu_c"]
    nu_c = material_params["nu_c"]

    return greens_displacement_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)


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
    Compute the Green's function for the response to a displacement force source in a 3D Cosserat medium.

    Parameters:
        x: np.ndarray
            The spatial location where the Green's function is evaluated. Should be a 3D vector.
        omega: float
            The angular frequency at which to evaluate the Green's function.
        rho: float
            Density of the medium.
        lam: float
            Lamé's first parameter.
        mu: float
            Shear modulus.
        nu: float
            Cosserat couple modulus.
        J: float
            Micro-inertia.
        lam_c: float
            Cosserat Lamé's first parameter.
        mu_c: float
            Cosserat shear modulus.
        nu_c: float
            Cosserat couple modulus.

    Returns:
        np.ndarray
            A 6x3 complex array representing the Green's function for response to a displacement force source.
            The indices correspond to (displacement component, rotation component).
    """

    if len(x) != 3:
        err = f"Spatial location x must have length 3 for 3D problems. Got length {len(x)}."
        raise ValueError(err)

    R = np.linalg.norm(x)
    if R == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)
    R_hat = x / R

    if omega == 0:
        return greens_displacement_force_static(
            x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        )

    # Compute squared velocities and cutoff frequency
    _, c2_sq, _, c4_sq = dispersion3d.all_c_squared(
        rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    w0_sq = dispersion3d.w0_squared(nu, J)

    # compute wavenumbers k1, k2, k3, k4 from the dispersion relation
    k1_sq, k2_sq, _, k4_sq = dispersion3d.all_k_squared(
        omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    k1, k2, _, k4 = dispersion3d.all_k(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    # define helper values
    A1 = 1 / (k2_sq - k1_sq) / (k4_sq - k1_sq)
    A2 = 1 / (k4_sq - k2_sq) / (k1_sq - k2_sq)
    A4 = 1 / (k1_sq - k4_sq) / (k2_sq - k4_sq)

    # -I_3 * \frac{\hat{f}_\omega}{r}\frac{\rho }{4\pi(\mu+\nu)}\frac{1}{k_4^2-k_2^2} *
    # \paren{e^{ik_2r}\paren{\frac{\omega^2-\omega_0^2}{c_4^2}-k_2^2}-e^{ik_4r}\paren{\frac{\omega^2-\omega_0^2}{c_4^2}-k_4^2}}
    term1_prefactor = 1 / (4 * np.pi * R * (mu + nu))
    # TODO: why not negative?
    term1 = (
        term1_prefactor
        * (
            (k4_sq - k2_sq) ** (-1)
            * (
                np.exp(1j * k2 * R) * ((omega**2 - w0_sq) / c4_sq - k2_sq)
                - np.exp(1j * k4 * R) * ((omega**2 - w0_sq) / c4_sq - k4_sq)
            )
        )
        * np.identity(3)
    )

    # I_3 * \frac{\rho\paren{\lambda+\mu-\nu}}{4\pi(\lambda+2\mu)(\mu+\nu)} *
    # \sum_{n=1,2,4} A_n \paren{\frac{\omega^2-\omega_0^2}{c_4^2}-\frac{j\omega_0^2}{4c_2^2c_4^2}-k_n^2}\paren{ik_nr-1}\frac{e^{ik_nr}}{r^3}
    term2_prefactor = -(lam + mu - nu) / (4 * np.pi * (lam + 2 * mu) * (mu + nu))
    # TODO: why negative?
    term2_n1 = (
        ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4 * c2_sq * c4_sq) - k1_sq)
        * (1j * k1 * R - 1)
        * np.exp(1j * k1 * R)
        / R**3
    )
    term2_n2 = (
        ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4 * c2_sq * c4_sq) - k2_sq)
        * (1j * k2 * R - 1)
        * np.exp(1j * k2 * R)
        / R**3
    )
    term2_n4 = (
        ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4 * c2_sq * c4_sq) - k4_sq)
        * (1j * k4 * R - 1)
        * np.exp(1j * k4 * R)
        / R**3
    )
    term2 = (
        term2_prefactor
        * (A1 * term2_n1 + A2 * term2_n2 + A4 * term2_n4)
        * np.identity(3)
    )

    # - \hat{r} \hat{r}^T \frac{\rho\paren{\lambda+\mu-\nu}}{4\pi(\lambda+2\mu)(\mu+\nu)} *
    # \sum_{n=1,2,4} A_n \paren{\frac{\omega^2-\omega_0^2}{c_4^2}-\frac{j\omega_0^2}{4c_2^2c_4^2}-k_n^2}\paren{-k_n^2r^2-3ik_nr+3} \frac{e^{ik_nr}}{r^3}
    term3_prefactor = -(lam + mu - nu) / (4 * np.pi * (lam + 2 * mu) * (mu + nu))
    term3_n1 = (
        ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4 * c2_sq * c4_sq) - k1_sq)
        * (-k1_sq * R**2 - 3j * k1 * R + 3)
        * np.exp(1j * k1 * R)
        / R**3
    )
    term3_n2 = (
        ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4 * c2_sq * c4_sq) - k2_sq)
        * (-k2_sq * R**2 - 3j * k2 * R + 3)
        * np.exp(1j * k2 * R)
        / R**3
    )
    term3_n4 = (
        ((omega**2 - w0_sq) / c4_sq - J * w0_sq / (4 * c2_sq * c4_sq) - k4_sq)
        * (-k4_sq * R**2 - 3j * k4 * R + 3)
        * np.exp(1j * k4 * R)
        / R**3
    )
    term3 = (
        term3_prefactor
        * (A1 * term3_n1 + A2 * term3_n2 + A4 * term3_n4)
        * np.outer(R_hat, R_hat)
    )

    # [\hat{r}\times -] \frac{1}{r^2} \frac{\rho \nu}{2\pi(\mu+\nu)(\mu_c+\nu_c)}\frac{1}{k_4^2-k_2^2}\paren{(ik_2r-1)e^{ik_2r}-(ik_4r-1)e^{ik_4r}}
    rotation_term_prefactor = -nu / (2 * np.pi * (mu + nu) * (mu_c + nu_c)) / R**2
    # TODO: why negative?
    rotation_term = (
        rotation_term_prefactor
        * (k4_sq - k2_sq) ** (-1)
        * (
            (1j * k2 * R - 1) * np.exp(1j * k2 * R)
            - (1j * k4 * R - 1) * np.exp(1j * k4 * R)
        )
        * np.array(
            [
                [0, -R_hat[2], R_hat[1]],
                [R_hat[2], 0, -R_hat[0]],
                [-R_hat[1], R_hat[0], 0],
            ]
        )
    )

    G = np.zeros((6, 3), dtype=np.complex128)
    G[:3, :] = term1 + term2 + term3
    G[3:, :] = rotation_term

    return G


def greens_displacement_force_static(
    x: np.ndarray,
    _rho: float,
    _lam: float,
    _mu: float,
    _nu: float,
    _J: float,
    _lam_c: float,
    _mu_c: float,
    _nu_c: float,
) -> np.ndarray:
    """Compute the Green's function for the response to a displacement force source in a 3D Cosserat medium at zero frequency (static case).

    Parameters:
        x: np.ndarray
            The spatial location where the Green's function is evaluated. Should be a 3D vector.
        rho: float
            Density of the medium.
        lam: float
            Lamé's first parameter.
        mu: float
            Shear modulus.
        nu: float
            Cosserat couple modulus.
        J: float
            Micro-inertia.
        lam_c: float
            Cosserat Lamé's first parameter.
        mu_c: float
            Cosserat shear modulus.
        nu_c: float
            Cosserat couple modulus.

    Returns:
        np.ndarray
            A 6x3 complex array representing the Green's function for response to a displacement force source.
    """
    # This needs to be derived from Eringen's book and for now is implemented as a zero response

    if len(x) != 3:
        err = f"Spatial location x must have length 3 for 3D problems. Got length {len(x)}."
        raise ValueError(err)

    R = np.linalg.norm(x)
    if R == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)
    _R_hat = x / R

    # TODO: derive the correct static green's function
    return np.zeros((6, 3), dtype=complex)  # divide by rho once we solve this


def greens_rotation_force_from_dict(
    x: np.ndarray,
    omega: float,
    material_params: dict,
) -> np.ndarray:
    """
    Compute the Green's function for the response to a rotation force source in a 3D Cosserat medium.

    Parameters:
        x: np.ndarray
            The spatial location where the Green's function is evaluated. Should be a 3D vector.
        omega: float
            The angular frequency at which to evaluate the Green's function.
        material_params: dict
            A dictionary containing the material parameters:
            - 'rho': Density
            - 'lam': Lamé's first parameter
            - 'mu': Shear modulus
            - 'nu': Cosserat couple modulus
            - 'J': Micro-inertia
            - 'lam_c': Cosserat Lamé's first parameter
            - 'mu_c': Cosserat shear modulus
            - 'nu_c': Cosserat couple modulus

    Returns:
        np.ndarray
            A 6x3 complex array representing the Green's function for response to a rotation force source.
            The indices correspond to (displacement component, rotation component).
    """

    # Extract material parameters
    rho = material_params["rho"]
    lam = material_params["lam"]
    mu = material_params["mu"]
    nu = material_params["nu"]
    J = material_params["J"]
    lam_c = material_params["lam_c"]
    mu_c = material_params["mu_c"]
    nu_c = material_params["nu_c"]

    return greens_rotation_force(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)


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
    """
    Compute the Green's function for the response to a rotation force source in a 3D Cosserat medium.

    Parameters:
        x: np.ndarray
            The spatial location where the Green's function is evaluated. Should be a 3D vector.
        omega: float
            The angular frequency at which to evaluate the Green's function.
        rho: float
            Density of the medium.
        lam: float
            Lamé's first parameter.
        mu: float
            Shear modulus.
        nu: float
            Cosserat couple modulus.
        J: float
            Micro-inertia.
        lam_c: float
            Cosserat Lamé's first parameter.
        mu_c: float
            Cosserat shear modulus.
        nu_c: float
            Cosserat couple modulus.

    Returns:
        np.ndarray
            A 6x3 complex array representing the Green's function for response to a rotation force source.
            The indices correspond to (displacement component, rotation component).
    """

    if len(x) != 3:
        err = f"Spatial location x must have length 3 for 3D problems. Got length {len(x)}."
        raise ValueError(err)

    R = np.linalg.norm(x)
    if R == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)
    R_hat = x / R

    if omega == 0:
        return greens_rotation_force_static(x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    # Compute squared velocities and cutoff frequency
    _, c2_sq, c3_sq, _ = dispersion3d.all_c_squared(
        rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    w0_sq = dispersion3d.w0_squared(nu, J)

    # compute wavenumbers k1, k2, k3, k4 from the dispersion relation
    _, k2_sq, k3_sq, k4_sq = dispersion3d.all_k_squared(
        omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    _, k2, k3, k4 = dispersion3d.all_k(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c)

    # define helper values
    B2 = 1 / (k3_sq - k2_sq) / (k4_sq - k2_sq)
    B3 = 1 / (k2_sq - k3_sq) / (k4_sq - k3_sq)
    B4 = 1 / (k2_sq - k4_sq) / (k3_sq - k4_sq)

    # [\hat{r} \times -]\frac{1}{r^2}\frac{\rho j \nu}{2\pi(\mu+\nu)(\mu_c+\nu_c)} *
    # \sum_{n=2,3,4}B_n \paren{\frac{\omega^2-\omega_0^2}{c_3^2}-k_n^2}\paren{ik_nr-1}e^{ik_nr}
    displacement_term_prefactor = (
        -J * nu / (2 * np.pi * (mu + nu) * (mu_c + nu_c)) / R**2
    )
    # TODO: why negative?
    displacement_term_n2 = (
        ((omega**2 - w0_sq) / c3_sq - k2_sq) * (1j * k2 * R - 1) * np.exp(1j * k2 * R)
    )
    displacement_term_n3 = (
        ((omega**2 - w0_sq) / c3_sq - k3_sq) * (1j * k3 * R - 1) * np.exp(1j * k3 * R)
    )
    displacement_term_n4 = (
        ((omega**2 - w0_sq) / c3_sq - k4_sq) * (1j * k4 * R - 1) * np.exp(1j * k4 * R)
    )
    displacement_term = (
        displacement_term_prefactor
        * (
            B2 * displacement_term_n2
            + B3 * displacement_term_n3
            + B4 * displacement_term_n4
        )
        * np.array(
            [
                [0, -R_hat[2], R_hat[1]],
                [R_hat[2], 0, -R_hat[0]],
                [-R_hat[1], R_hat[0], 0],
            ]
        )
    )

    # -I_3\frac{1}{r} \frac{\rho j}{4\pi(\mu_c+\nu_c)}\sum_{n=2,3,4}B_n \paren{\frac{\omega^2}{c_2^2}-k_n^2} *
    # \paren{\frac{\omega^2-\omega_0^2}{c_3^2}-k_n^2}e^{ik_nr}
    term1_prefactor = J / (4 * np.pi * R * (mu_c + nu_c))
    # TODO: why not negative?
    term1_n2 = (
        (omega**2 / c2_sq - k2_sq)
        * ((omega**2 - w0_sq) / c3_sq - k2_sq)
        * np.exp(1j * k2 * R)
    )
    term1_n3 = (
        (omega**2 / c2_sq - k3_sq)
        * ((omega**2 - w0_sq) / c3_sq - k3_sq)
        * np.exp(1j * k3 * R)
    )
    term1_n4 = (
        (omega**2 / c2_sq - k4_sq)
        * ((omega**2 - w0_sq) / c3_sq - k4_sq)
        * np.exp(1j * k4 * R)
    )
    term1 = (
        term1_prefactor
        * (B2 * term1_n2 + B3 * term1_n3 + B4 * term1_n4)
        * np.identity(3)
    )

    # I_3\frac{\rho j(\lambda_c+\mu_c-\nu_c)}{4\pi(\lambda_c+2\mu_c)(\mu_c+\nu_c)} *
    # \sum_{n=2,3,4}B_n\paren{\frac{\omega^2}{c_2^2}-k_n^2}\paren{ik_nr-1}\frac{e^{ik_nr}}{r^3}
    term2_prefactor = (
        -J * (lam_c + mu_c - nu_c) / (4 * np.pi * (lam_c + 2 * mu_c) * (mu_c + nu_c))
    )
    # TODO: why negative?
    term2_n2 = (
        (omega**2 / c2_sq - k2_sq) * (1j * k2 * R - 1) * np.exp(1j * k2 * R) / R**3
    )
    term2_n3 = (
        (omega**2 / c2_sq - k3_sq) * (1j * k3 * R - 1) * np.exp(1j * k3 * R) / R**3
    )
    term2_n4 = (
        (omega**2 / c2_sq - k4_sq) * (1j * k4 * R - 1) * np.exp(1j * k4 * R) / R**3
    )
    term2 = (
        term2_prefactor
        * (B2 * term2_n2 + B3 * term2_n3 + B4 * term2_n4)
        * np.identity(3)
    )

    # -I_3\frac{\rho j\nu^2}{\pi(\lambda_c+2\mu_c)(\mu+\nu)(\mu_c+\nu_c)} *
    # \sum_{n=2,3,4}B_n \paren{ik_nr-1}\frac{e^{ik_nr}}{r^3}
    term3_prefactor = (
        J * nu**2 / (np.pi * (lam_c + 2 * mu_c) * (mu + nu) * (mu_c + nu_c))
    )
    # TODO: why not negative?
    term3_n2 = (1j * k2 * R - 1) * np.exp(1j * k2 * R) / R**3
    term3_n3 = (1j * k3 * R - 1) * np.exp(1j * k3 * R) / R**3
    term3_n4 = (1j * k4 * R - 1) * np.exp(1j * k4 * R) / R**3
    term3 = (
        term3_prefactor
        * (B2 * term3_n2 + B3 * term3_n3 + B4 * term3_n4)
        * np.identity(3)
    )

    # - \hat{r}\hat{r}^T \frac{\rho j(\lambda_c+\mu_c-\nu_c)}{4\pi(\lambda_c+2\mu_c)(\mu_c+\nu_c)} *
    # \sum_{n=2,3,4} B_n  \paren{\frac{\omega^2}{c_2^2}-k_n^2}\paren{-k_n^2r^2-3ik_nr+3} \frac{e^{ik_nr}}{r^3}
    term4_prefactor = (
        -J * (lam_c + mu_c - nu_c) / (4 * np.pi * (lam_c + 2 * mu_c) * (mu_c + nu_c))
    )
    term4_n2 = (
        (omega**2 / c2_sq - k2_sq)
        * (-k2_sq * R**2 - 3j * k2 * R + 3)
        * np.exp(1j * k2 * R)
        / R**3
    )
    term4_n3 = (
        (omega**2 / c2_sq - k3_sq)
        * (-k3_sq * R**2 - 3j * k3 * R + 3)
        * np.exp(1j * k3 * R)
        / R**3
    )
    term4_n4 = (
        (omega**2 / c2_sq - k4_sq)
        * (-k4_sq * R**2 - 3j * k4 * R + 3)
        * np.exp(1j * k4 * R)
        / R**3
    )
    term4 = (
        term4_prefactor
        * (B2 * term4_n2 + B3 * term4_n3 + B4 * term4_n4)
        * np.outer(R_hat, R_hat)
    )

    # \hat{r}\hat{r}^T \frac{\rho j\nu^2}{\pi(\lambda_c+2\mu_c)(\mu+\nu)(\mu_c+\nu_c)} *
    # \sum_{n=2,3,4} B_n \paren{-k_n^2r^2-3ik_nr+3} \frac{e^{ik_nr}}{r^3}
    term5_prefactor = (
        J * nu**2 / (np.pi * (lam_c + 2 * mu_c) * (mu + nu) * (mu_c + nu_c))
    )
    term5_n2 = (-k2_sq * R**2 - 3j * k2 * R + 3) * np.exp(1j * k2 * R) / R**3
    term5_n3 = (-k3_sq * R**2 - 3j * k3 * R + 3) * np.exp(1j * k3 * R) / R**3
    term5_n4 = (-k4_sq * R**2 - 3j * k4 * R + 3) * np.exp(1j * k4 * R) / R**3
    term5 = (
        term5_prefactor
        * (B2 * term5_n2 + B3 * term5_n3 + B4 * term5_n4)
        * np.outer(R_hat, R_hat)
    )

    G = np.zeros((6, 3), dtype=np.complex128)
    G[:3, :] = displacement_term
    G[3:, :] = term1 + term2 + term3 + term4 + term5

    return G / J  # TODO: why J and not rho?


def greens_rotation_force_static(
    x: np.ndarray,
    _rho: float,
    _lam: float,
    _mu: float,
    _nu: float,
    _J: float,
    _lam_c: float,
    _mu_c: float,
    _nu_c: float,
) -> np.ndarray:
    """Compute the Green's function for the response to a rotation force source in a 3D Cosserat medium at zero frequency (static case).

    Parameters:
        x: np.ndarray
            The spatial location where the Green's function is evaluated. Should be a 3D vector.
        rho: float
            Density of the medium.
        lam: float
            Lamé's first parameter.
        mu: float
            Shear modulus.
        nu: float
            Cosserat couple modulus.
        J: float
            Micro-inertia.
        lam_c: float
            Cosserat Lamé's first parameter.
        mu_c: float
            Cosserat shear modulus.
        nu_c: float
            Cosserat couple modulus.

    Returns:
        np.ndarray
            A 6x3 complex array representing the Green's function for response to a rotation force source.
    """
    # This needs to be derived from Eringen's book and for now is implemented as a zero response

    if len(x) != 3:
        err = f"Spatial location x must have length 3 for 3D problems. Got length {len(x)}."
        raise ValueError(err)

    R = np.linalg.norm(x)
    if R == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)
    _R_hat = x / R

    # TODO: derive the correct static green's function
    return np.zeros((6, 3), dtype=complex)  # divide by rho once we solve this
