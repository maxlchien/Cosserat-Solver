from __future__ import annotations

try:
    from cosserat_solver import cosserat_core_3d

    HAS_FORTRAN = True
except ImportError:
    HAS_FORTRAN = False

import numpy as np


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

    if np.linalg.norm(x) == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)

    return cosserat_core_3d.greens_mixed_force(
        x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )


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

    if np.linalg.norm(x) == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)

    return cosserat_core_3d.greens_displacement_force(
        x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )


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

    if np.linalg.norm(x) == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)

    return cosserat_core_3d.greens_displacement_force_static(
        x, _rho, _lam, _mu, _nu, _J, _lam_c, _mu_c, _nu_c
    )


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

    if np.linalg.norm(x) == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)

    return cosserat_core_3d.greens_rotation_force(
        x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )


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

    if np.linalg.norm(x) == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)

    return cosserat_core_3d.greens_rotation_force_static(
        x, _rho, _lam, _mu, _nu, _J, _lam_c, _mu_c, _nu_c
    )
