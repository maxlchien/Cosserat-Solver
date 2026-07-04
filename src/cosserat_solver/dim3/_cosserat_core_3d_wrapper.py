from __future__ import annotations

try:
    from cosserat_solver.dim3 import cosserat_core_3d

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

    return np.array(
        cosserat_core_3d.greens_mixed_force(
            x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        )
    )


def greens_mixed_force_vectorized(
    x: np.ndarray,
    omega: float | np.ndarray,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
    force_use_openmp: bool = False,
    force_no_openmp: bool = False,
) -> np.ndarray:
    """
    Compute the Green's function for the response to a mixed force source in a 3D Cosserat medium.

    Automatically detects whether omega is scalar or array and handles appropriately.
    Scalar inputs are converted to length-1 arrays internally for unified processing.

    Parameters
    ----------
    x : np.ndarray
        3D position vector [x, y, z]
    omega : float | np.ndarray
        Angular frequency (scalar) or array of angular frequencies
    rho : float
        Density of the medium
    lam : float
        Lamé's first parameter
    mu : float
        Shear modulus
    nu : float
        Cosserat couple modulus
    J : float
        Micro-inertia
    lam_c : float
        Cosserat Lamé's first parameter
    mu_c : float
        Cosserat shear modulus
    nu_c : float
        Cosserat couple modulus
    force_use_openmp : bool, default=False
        If True, force OpenMP parallelization even for small arrays.
        Mutually exclusive with force_no_openmp.
    force_no_openmp : bool, default=False
        If True, disable OpenMP parallelization even for large arrays.
        Mutually exclusive with force_use_openmp.

    Returns
    -------
    np.ndarray
        If omega is scalar: shape (6, 6) complex array
        If omega is array: shape (n_omega, 6, 6) complex array

    Raises
    ------
    ValueError
        If both force_use_openmp and force_no_openmp are True
    """

    # Validate mutual exclusivity
    if force_use_openmp and force_no_openmp:
        err = "force_use_openmp and force_no_openmp are mutually exclusive"
        raise ValueError(err)

    # Detect if omega is scalar or array
    if np.isscalar(omega):
        # Scalar case - wrap in array
        omega_array = np.array([float(omega)], dtype=float)
        squeeze_output = True
    else:
        # Array case
        omega_array = np.asarray(omega, dtype=float)
        if omega_array.ndim != 1:
            err = "omega must be scalar or 1D array"
            raise ValueError(err)
        squeeze_output = False

    # Call vectorized Fortran backend
    # Returns list of tuples (one per omega)
    result_list = cosserat_core_3d.greens_mixed_force_vectorized(
        x,
        omega_array,
        rho,
        lam,
        mu,
        nu,
        J,
        lam_c,
        mu_c,
        nu_c,
        int(force_use_openmp),
        int(force_no_openmp),
    )

    # Convert list of tuples to numpy array
    n_omega = len(result_list)
    result_array = np.zeros((n_omega, 6, 6), dtype=np.complex128)
    for i, matrix_tuple in enumerate(result_list):
        result_array[i] = np.array(matrix_tuple, dtype=np.complex128)

    # Return scalar result if input was scalar
    if squeeze_output:
        return result_array[0]
    return result_array


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

    return np.array(
        cosserat_core_3d.greens_displacement_force(
            x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        )
    )


def greens_displacement_force_vectorized(
    x: np.ndarray,
    omega: float | np.ndarray,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
    force_use_openmp: bool = False,
    force_no_openmp: bool = False,
) -> np.ndarray:
    """
    Compute the Green's function for the response to a displacement force source in a 3D Cosserat medium.

    Automatically detects whether omega is scalar or array and handles appropriately.
    Scalar inputs are converted to length-1 arrays internally for unified processing.

    Parameters
    ----------
    x : np.ndarray
        3D position vector [x, y, z]
    omega : float | np.ndarray
        Angular frequency (scalar) or array of angular frequencies
    rho : float
        Density of the medium
    lam : float
        Lamé's first parameter
    mu : float
        Shear modulus
    nu : float
        Cosserat couple modulus
    J : float
        Micro-inertia
    lam_c : float
        Cosserat Lamé's first parameter
    mu_c : float
        Cosserat shear modulus
    nu_c : float
        Cosserat couple modulus
    force_use_openmp : bool, default=False
        If True, force OpenMP parallelization even for small arrays.
        Mutually exclusive with force_no_openmp.
    force_no_openmp : bool, default=False
        If True, disable OpenMP parallelization even for large arrays.
        Mutually exclusive with force_use_openmp.

    Returns
    -------
    np.ndarray
        If omega is scalar: shape (6, 3) complex array
        If omega is array: shape (n_omega, 6, 3) complex array

    Raises
    ------
    ValueError
        If both force_use_openmp and force_no_openmp are True
    """

    # Validate mutual exclusivity
    if force_use_openmp and force_no_openmp:
        err = "force_use_openmp and force_no_openmp are mutually exclusive"
        raise ValueError(err)

    # Detect if omega is scalar or array
    if np.isscalar(omega):
        # Scalar case - wrap in array
        omega_array = np.array([float(omega)], dtype=float)
        squeeze_output = True
    else:
        # Array case
        omega_array = np.asarray(omega, dtype=float)
        if omega_array.ndim != 1:
            err = "omega must be scalar or 1D array"
            raise ValueError(err)
        squeeze_output = False

    # Call vectorized Fortran backend
    # Returns list of tuples (one per omega)
    result_list = cosserat_core_3d.greens_displacement_force_vectorized(
        x,
        omega_array,
        rho,
        lam,
        mu,
        nu,
        J,
        lam_c,
        mu_c,
        nu_c,
        int(force_use_openmp),
        int(force_no_openmp),
    )

    # Convert list of tuples to numpy array
    n_omega = len(result_list)
    result_array = np.zeros((n_omega, 6, 3), dtype=np.complex128)
    for i, matrix_tuple in enumerate(result_list):
        result_array[i] = np.array(matrix_tuple, dtype=np.complex128)

    # Return scalar result if input was scalar
    if squeeze_output:
        return result_array[0]
    return result_array


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

    return np.array(
        cosserat_core_3d.greens_displacement_force_static(
            x, _rho, _lam, _mu, _nu, _J, _lam_c, _mu_c, _nu_c
        )
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

    return np.array(
        cosserat_core_3d.greens_rotation_force(
            x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
        )
    )


def greens_rotation_force_vectorized(
    x: np.ndarray,
    omega: float | np.ndarray,
    rho: float,
    lam: float,
    mu: float,
    nu: float,
    J: float,
    lam_c: float,
    mu_c: float,
    nu_c: float,
    force_use_openmp: bool = False,
    force_no_openmp: bool = False,
) -> np.ndarray:
    """
    Compute the Green's function for the response to a rotation force source in a 3D Cosserat medium.

    Automatically detects whether omega is scalar or array and handles appropriately.
    Scalar inputs are converted to length-1 arrays internally for unified processing.

    Parameters
    ----------
    x : np.ndarray
        3D position vector [x, y, z]
    omega : float | np.ndarray
        Angular frequency (scalar) or array of angular frequencies
    rho : float
        Density of the medium
    lam : float
        Lamé's first parameter
    mu : float
        Shear modulus
    nu : float
        Cosserat couple modulus
    J : float
        Micro-inertia
    lam_c : float
        Cosserat Lamé's first parameter
    mu_c : float
        Cosserat shear modulus
    nu_c : float
        Cosserat couple modulus
    force_use_openmp : bool, default=False
        If True, force OpenMP parallelization even for small arrays.
        Mutually exclusive with force_no_openmp.
    force_no_openmp : bool, default=False
        If True, disable OpenMP parallelization even for large arrays.
        Mutually exclusive with force_use_openmp.

    Returns
    -------
    np.ndarray
        If omega is scalar: shape (6, 3) complex array
        If omega is array: shape (n_omega, 6, 3) complex array

    Raises
    ------
    ValueError
        If both force_use_openmp and force_no_openmp are True
    """

    # Validate mutual exclusivity
    if force_use_openmp and force_no_openmp:
        err = "force_use_openmp and force_no_openmp are mutually exclusive"
        raise ValueError(err)

    # Detect if omega is scalar or array
    if np.isscalar(omega):
        # Scalar case - wrap in array
        omega_array = np.array([float(omega)], dtype=float)
        squeeze_output = True
    else:
        # Array case
        omega_array = np.asarray(omega, dtype=float)
        if omega_array.ndim != 1:
            err = "omega must be scalar or 1D array"
            raise ValueError(err)
        squeeze_output = False

    # Call vectorized Fortran backend
    # Returns list of tuples (one per omega)
    result_list = cosserat_core_3d.greens_rotation_force_vectorized(
        x,
        omega_array,
        rho,
        lam,
        mu,
        nu,
        J,
        lam_c,
        mu_c,
        nu_c,
        int(force_use_openmp),
        int(force_no_openmp),
    )

    # Convert list of tuples to numpy array
    n_omega = len(result_list)
    result_array = np.zeros((n_omega, 6, 3), dtype=np.complex128)
    for i, matrix_tuple in enumerate(result_list):
        result_array[i] = np.array(matrix_tuple, dtype=np.complex128)

    # Return scalar result if input was scalar
    if squeeze_output:
        return result_array[0]
    return result_array


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

    return np.array(
        cosserat_core_3d.greens_rotation_force_static(
            x, _rho, _lam, _mu, _nu, _J, _lam_c, _mu_c, _nu_c
        )
    )
