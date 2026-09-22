from __future__ import annotations

try:
    from cosserat_solver.dim3 import elastic_core

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

    G = np.zeros((6, 6), dtype=np.complex128)
    G[:, :3] = greens_displacement_force(
        x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
    )
    return G


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
        If omega is scalar: shape (3, 3) complex array
        If omega is array: shape (n_omega, 3, 3) complex array

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

    disp_block = greens_displacement_force_vectorized(
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
        force_use_openmp,
        force_no_openmp,
    )

    n_omega = len(omega_array)

    # Return scalar result if input was scalar
    if squeeze_output:
        G = np.zeros((6, 6), dtype=np.complex128)
        G[:, :3] = disp_block[0]
        return G
    G = np.zeros((n_omega, 6, 6), dtype=np.complex128)
    G[:, :, :3] = disp_block
    return G


def greens_mixed_force_points_vectorized(
    points: np.ndarray,
    omega: np.ndarray,
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
    Compute the mixed-force Green's function over a batch of positions and frequencies
    for a 3D elastic medium.

    The elastic backend evaluates a 3x3 displacement-force block per (point, frequency);
    it is embedded into the top-left 3x3 corner of a zero-filled 6x6 tensor to match the
    Cosserat mixed-force convention. The Cosserat-only parameters are accepted for a
    uniform signature and ignored.

    Parameters
    ----------
    points : np.ndarray
        Positions at which to evaluate, shape (n_points, 3).
    omega : np.ndarray
        Angular frequencies, shape (n_omega,).
    rho : float
        Density of the medium
    lam : float
        Lamé's first parameter
    mu : float
        Shear modulus
    nu, J, lam_c, mu_c, nu_c : float
        Cosserat parameters, unused for the elastic case.
    force_use_openmp : bool, default=False
        If True, force OpenMP parallelization even for small arrays.
        Mutually exclusive with force_no_openmp.
    force_no_openmp : bool, default=False
        If True, disable OpenMP parallelization even for large arrays.
        Mutually exclusive with force_use_openmp.

    Returns
    -------
    np.ndarray
        Complex array of shape (n_points, n_omega, 6, 6).

    Raises
    ------
    ValueError
        If both force_use_openmp and force_no_openmp are True, or if the input
        shapes are invalid.
    """
    _ = (nu, J, lam_c, mu_c, nu_c)  # Unused parameters for the elastic case

    # Validate mutual exclusivity
    if force_use_openmp and force_no_openmp:
        err = "force_use_openmp and force_no_openmp are mutually exclusive"
        raise ValueError(err)

    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        err = "points must have shape (n_points, 3)"
        raise ValueError(err)

    omega_array = np.ascontiguousarray(omega, dtype=np.float64)
    if omega_array.ndim != 1:
        err = "omega must be a 1D array"
        raise ValueError(err)
    if omega_array.size < 1:
        err = "omega must contain at least one frequency"
        raise ValueError(err)

    n_points = points.shape[0]
    n_omega = omega_array.shape[0]

    # Caller-allocated C-order buffers. The Fortran output
    # result_real(n_omega, 3, 3, n_points) in column-major order is byte-identical
    # to a C-order array of shape (n_points, 3, 3, n_omega) indexed [p, col, row, w].
    out_real = np.empty((n_points, 3, 3, n_omega), dtype=np.float64)
    out_imag = np.empty((n_points, 3, 3, n_omega), dtype=np.float64)

    elastic_core.greens_displacement_force_points_vectorized(
        points,
        omega_array,
        rho,
        lam,
        mu,
        int(force_use_openmp),
        int(force_no_openmp),
        out_real,
        out_imag,
    )

    # [p, col, row, w] -> [p, w, row, col]  (3x3 displacement block)
    disp_block = (out_real + 1j * out_imag).transpose(0, 3, 2, 1)

    # Embed the 3x3 displacement block into the top-left corner of a 6x6 tensor,
    # matching greens_mixed_force_vectorized's G[:, :3, :3] = disp convention.
    G = np.zeros((n_points, n_omega, 6, 6), dtype=np.complex128)
    G[:, :, :3, :3] = disp_block
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
    _ = (nu, J, lam_c, mu_c, nu_c)  # Unused parameters for the elastic case

    if len(x) != 3:
        err = f"Spatial location x must have length 3 for 3D problems. Got length {len(x)}."
        raise ValueError(err)

    if np.linalg.norm(x) == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)

    G = np.zeros((6, 3), dtype=np.complex128)
    G[:3, :] = elastic_core.greens_displacement_force(x, omega, rho, lam, mu)
    return G


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
    Compute the Green's function for the response to a displacement force source in a 3D elastic medium.

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
    _ = (nu, J, lam_c, mu_c, nu_c)  # Unused parameters for the elastic case

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
    result_list = elastic_core.greens_displacement_force_vectorized(
        x,
        omega_array,
        rho,
        lam,
        mu,
        int(force_use_openmp),
        int(force_no_openmp),
    )

    # Convert list of tuples to numpy array
    n_omega = len(result_list)
    result_array = np.zeros((n_omega, 3, 3), dtype=np.complex128)
    for i, matrix_tuple in enumerate(result_list):
        result_array[i] = np.array(matrix_tuple, dtype=np.complex128)

    # Return scalar result if input was scalar
    if squeeze_output:
        G = np.zeros((6, 3), dtype=np.complex128)
        G[:3, :] = result_array[0]
        return G
    G = np.zeros((n_omega, 6, 3), dtype=np.complex128)
    G[:, :3, :] = result_array
    return G


def greens_rotation_force_from_dict(
    x: np.ndarray,
    omega: float,
    material_params: dict,
) -> np.ndarray:
    """
    Compute the Green's function for the response to a rotation force source in a 3D elastic medium (always zero).

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
    Compute the Green's function for the response to a rotation force source in a 3D elastic medium (always zero).

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

    _ = (
        x,
        omega,
        rho,
        lam,
        mu,
        nu,
        J,
        lam_c,
        mu_c,
        nu_c,
    )  # Unused parameters for the elastic case

    if len(x) != 3:
        err = f"Spatial location x must have length 3 for 3D problems. Got length {len(x)}."
        raise ValueError(err)

    if np.linalg.norm(x) == 0:
        err = "Spatial location x cannot be the zero vector for Green's function evaluation."
        raise ValueError(err)

    return np.zeros(
        (6, 3), dtype=np.complex128
    )  # No response for rotation force in the elastic case


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
    Compute the Green's function for the response to a rotation force source in a 3D elastic medium (always zero).

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
        If omega is scalar: shape (3, 3) complex array (all zeros)
        If omega is array: shape (n_omega, 3, 3) complex array (all zeros)

    Raises
    ------
    ValueError
        If both force_use_openmp and force_no_openmp are True
    """
    _ = (
        x,
        rho,
        lam,
        mu,
        nu,
        J,
        lam_c,
        mu_c,
        nu_c,
    )  # all parameters unused for rotation

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

    # Return zero array since rotation force has no response in the elastic case
    # Squeeze output if omega was scalar
    n_omega = len(omega_array)
    if squeeze_output:
        return np.zeros((6, 3), dtype=np.complex128)
    return np.zeros((n_omega, 6, 3), dtype=np.complex128)
