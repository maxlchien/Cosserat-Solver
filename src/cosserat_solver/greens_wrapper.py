"""
Unified wrapper for Green's function evaluation.

Provides a consistent interface to compute Green's functions using either:
- Fortran backend (fast, vectorized, lower precision)
- Python backend (slow, scalar, arbitrary precision)

The Fortran backend is preferred when available and provides significant speedup.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

try:
    import cosserat_solver.dim3._cosserat_core_wrapper as cosserat_wrapper
    import cosserat_solver.dim3._elastic_core_wrapper as elastic_wrapper
    from cosserat_solver.dim2._integrator_core_wrapper import (
        HAS_FORTRAN,
        IntegratorFortran,
    )

    FORTRAN_AVAILABLE = HAS_FORTRAN
except ImportError:
    FORTRAN_AVAILABLE = False
    if TYPE_CHECKING:
        from cosserat_solver.dim2._integrator_core_wrapper import IntegratorFortran
    warnings.warn("Fortran backend not available, using Python fallback", stacklevel=2)

from loguru import logger

import cosserat_solver.dim3.cosserat as cosserat
import cosserat_solver.dim3.elastic as elastic
from cosserat_solver import consts
from cosserat_solver.dim2.integrator import Integrator
from cosserat_solver.source import SourceSpectrum

# material type tags
MATERIAL_TYPE_ELASTIC = 0
MATERIAL_TYPE_COSSERAT = 1


def get_material_tag(material_type: str) -> int:
    if material_type.lower() == "elastic":
        return MATERIAL_TYPE_ELASTIC
    if material_type.lower() == "cosserat":
        return MATERIAL_TYPE_COSSERAT

    msg = f"Unknown material type: {material_type}. Supported types are 'elastic' and 'cosserat'."
    logger.error(msg)
    raise ValueError(msg)


# dimension tags
DIMENSION_2D = 2
DIMENSION_3D = 3

# backend tags
BACKEND_FORTRAN = 0
BACKEND_PYTHON = 1

SUPPORTED_COMBOS = {
    (DIMENSION_2D, BACKEND_FORTRAN, MATERIAL_TYPE_COSSERAT),
    (DIMENSION_2D, BACKEND_PYTHON, MATERIAL_TYPE_COSSERAT),
    (DIMENSION_3D, BACKEND_FORTRAN, MATERIAL_TYPE_COSSERAT),
    (DIMENSION_3D, BACKEND_PYTHON, MATERIAL_TYPE_COSSERAT),
    (DIMENSION_3D, BACKEND_FORTRAN, MATERIAL_TYPE_ELASTIC),
    (DIMENSION_3D, BACKEND_PYTHON, MATERIAL_TYPE_ELASTIC),
}


def _validate_material_params(material_params: dict) -> None:
    """Validate that all required material parameters are present."""
    required_keys = ["rho", "lam", "mu", "nu", "J", "lam_c", "mu_c", "nu_c"]
    missing = [key for key in required_keys if key not in material_params]
    if missing:
        err = f"Missing material parameters: {missing}"
        logger.error(err)
        raise ValueError(err)


def _validate_dimension_backend_material_combo(
    dim: int, backend: int, material_type: int
) -> None:
    if (dim, backend, material_type) not in SUPPORTED_COMBOS:
        err = f"Combination of dimension {dim}, backend {backend}, and material type {material_type} is not supported."
        logger.error(err)
        raise ValueError(err)


def evaluate_greens_fortran(
    x: np.ndarray,
    dim: int,
    omega_array: np.ndarray,
    material_params: dict,
    material_type: int = MATERIAL_TYPE_COSSERAT,
    force_use_openmp: bool = False,
    force_no_openmp: bool = False,
) -> np.ndarray:
    """
    Evaluate Green's function at multiple omega points using Fortran backend.

    Parameters
    ----------
    x : np.ndarray
        2D spatial location [x1, x2] where Green's function is evaluated
    dim: int
        The dimension of the problem (either 2 or 3)
    omega_array : np.ndarray
        Array of angular frequencies (real values)
    material_params : dict
        Material parameters (rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    force_use_openmp : bool, default=False
        If True, force OpenMP parallelization even for small arrays.
    force_no_openmp : bool, default=False
        If True, disable OpenMP parallelization even for large arrays.

    Returns
    -------
    greens : np.ndarray
        Complex array of shape (len(omega_array), 3, 3)
        Green's function evaluated at each omega

    Raises
    ------
    ValueError
        If both force_use_openmp and force_no_openmp are True
    """
    if not FORTRAN_AVAILABLE:
        err = "Fortran backend not available."
        logger.error(err)
        raise RuntimeError(err)

    _validate_material_params(material_params)
    _validate_dimension_backend_material_combo(dim, BACKEND_FORTRAN, material_type)
    x = np.asarray(x, dtype=float)
    if dim == 2 and material_type == MATERIAL_TYPE_COSSERAT:
        # Create Fortran integrator
        integrator = IntegratorFortran(
            rho=material_params["rho"],
            lam=material_params["lam"],
            mu=material_params["mu"],
            nu=material_params["nu"],
            J=material_params["J"],
            lam_c=material_params["lam_c"],
            mu_c=material_params["mu_c"],
            nu_c=material_params["nu_c"],
        )

        # Ensure x is a list/tuple of length 2
        if x.shape != (2,):
            err = "Spatial location x must have shape (2,) for dimension 2."
            logger.error(err)
            raise ValueError(err)

        # Use vectorized backend - automatically handles array
        return integrator.greens_x_omega_vectorized(
            x.tolist(),  # Convert to list for Fortran compatibility
            omega_array,
            force_use_openmp=force_use_openmp,
            force_no_openmp=force_no_openmp,
        )
    if dim == 3 and material_type == MATERIAL_TYPE_ELASTIC:
        # For 3D elastic, use the elastic_wrapper
        rho = material_params["rho"]
        lam = material_params["lam"]
        mu = material_params["mu"]

        # Ensure x is a list/tuple of length 3
        if x.shape != (3,):
            err = "Spatial location x must have shape (3,) for dimension 3."
            logger.error(err)
            raise ValueError(err)

        return elastic_wrapper.greens_mixed_force_vectorized(
            x,
            omega_array,
            rho,
            lam,
            mu,
            0,
            1,
            0,
            0,
            0,  # dummy values for nu, J, ... due to API
            force_use_openmp=force_use_openmp,
            force_no_openmp=force_no_openmp,
        )  # shape (n_omega, 6, 6)

    if dim == 3 and material_type == MATERIAL_TYPE_COSSERAT:
        # For 3D Cosserat, use the cosserat_wrapper
        rho = material_params["rho"]
        lam = material_params["lam"]
        mu = material_params["mu"]
        nu = material_params["nu"]
        J = material_params["J"]
        lam_c = material_params["lam_c"]
        mu_c = material_params["mu_c"]
        nu_c = material_params["nu_c"]

        # Ensure x is a list/tuple of length 3
        if x.shape != (3,):
            err = "Spatial location x must have shape (3,) for dimension 3."
            logger.error(err)
            raise ValueError(err)

        return cosserat_wrapper.greens_mixed_force_vectorized(
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
            force_use_openmp=force_use_openmp,
            force_no_openmp=force_no_openmp,
        )  # shape (n_omega, 6, 6)
    err = f"Combination of dimension {dim} and material type {material_type} is not supported for Fortran backend."
    logger.error(err)
    raise ValueError(err)


def evaluate_greens_python(
    x: np.ndarray,
    dim: int,
    omega: float,
    material_params: dict,
    material_type: int = MATERIAL_TYPE_COSSERAT,
    digits_precision: int = consts.COMPUTE_PRECISION,
) -> np.ndarray:
    """
    Evaluate Green's function at a single omega using Python/mpmath backend.

    Parameters
    ----------
    x : np.ndarray
        Spatial location where Green's function is evaluated
    dim: int
        The dimension of the problem (either 2 or 3)
    omega : float
        Angular frequency (scalar value)
    material_params : dict
        Material parameters (rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    digits_precision : int
        Number of digits for mpmath precision

    Returns
    -------
    greens : np.ndarray
        Complex array of shape (3, 3) for 2D or (6, 6) for 3D
        Green's function at omega
    """
    _validate_material_params(material_params)
    _validate_dimension_backend_material_combo(dim, BACKEND_PYTHON, material_type)
    x = np.asarray(x, dtype=float)

    if dim == 2:
        # Create Python integrator
        integrator = Integrator(
            rho=material_params["rho"],
            lam=material_params["lam"],
            mu=material_params["mu"],
            nu=material_params["nu"],
            J=material_params["J"],
            lam_c=material_params["lam_c"],
            mu_c=material_params["mu_c"],
            nu_c=material_params["nu_c"],
            digits_precision=digits_precision,
        )

        return integrator.greens_x_omega(x, omega)
    err = f"Invalid dimension {dim}. This function is only for numerics comparisons, and needs to be rewritten for 3D."
    logger.error(err)
    raise NotImplementedError(err)


def get_greens_callback(
    x: np.ndarray,
    dim: int,
    material_params: dict,
    source: SourceSpectrum,
    use_fortran: bool,
    material_type: int,
    digits_precision: int = consts.COMPUTE_PRECISION,
    force_use_openmp: bool = False,
    force_no_openmp: bool = False,
) -> Callable:
    """
    Returns a callback function for evaluating Green's function spectrum.

    The returned function can be used with the Fourier transform routines.
    It automatically chooses between Fortran (vectorized) and Python (scalar) backends.

    Parameters
    ----------
    x : np.ndarray
        2D spatial location where Green's function is evaluated
    dim: int
        The dimension of the problem (either 2 or 3)
    material_params : dict
        Material parameters (rho, lam, mu, nu, J, lam_c, mu_c, nu_c)
    source : SourceSpectrum
        Source object with spectrum(omega) and direction() methods
    use_fortran : bool
        Whether to use Fortran backend if available. Falls back to Python if not.
    material_type : int
        Type of material (e.g., MATERIAL_TYPE_COSSERAT)
    digits_precision : int
        Number of digits for mpmath precision (Python backend only)
    force_use_openmp : bool, default=False
        If True, force OpenMP parallelization even for small arrays.
    force_no_openmp : bool, default=False
        If True, disable OpenMP parallelization even for large arrays.

    Returns
    -------
    callback : Callable
        Function that takes omega (scalar or array) and returns spectrum.
        - If Fortran: accepts np.ndarray, returns shape (N, 3) array
        - If Python: accepts scalar float, returns shape (3,) array

    Notes
    -----
    The Fortran backend is significantly faster.
    The Python backend is much slower.
    """
    _validate_material_params(material_params)
    _validate_dimension_backend_material_combo(
        dim, BACKEND_FORTRAN if use_fortran else BACKEND_PYTHON, material_type
    )
    x = np.asarray(x, dtype=float)
    # Try to use Fortran backend
    if use_fortran and FORTRAN_AVAILABLE:
        if dim == 2 and material_type == MATERIAL_TYPE_COSSERAT:
            # Create integrator once for reuse
            try:
                integrator_fortran = IntegratorFortran(
                    rho=material_params["rho"],
                    lam=material_params["lam"],
                    mu=material_params["mu"],
                    nu=material_params["nu"],
                    J=material_params["J"],
                    lam_c=material_params["lam_c"],
                    mu_c=material_params["mu_c"],
                    nu_c=material_params["nu_c"],
                )
                x_2d = [float(x[0]), float(x[1])]

                def fortran_callback(omega_array: np.ndarray) -> np.ndarray:
                    """
                    Vectorized evaluation using Fortran backend.

                    Parameters
                    ----------
                    omega_array : np.ndarray
                        Array of angular frequencies

                    Returns
                    -------
                    spectrum : np.ndarray
                        Shape (N, 3, 3) array of Green's function spectrum times source magnitude
                    """
                    if np.isscalar(omega_array):
                        omega_array = np.array([omega_array])
                        squeeze_output = True
                    else:
                        squeeze_output = False

                    # Use vectorized backend with OpenMP control flags
                    G_omega = integrator_fortran.greens_x_omega_vectorized(
                        x_2d,
                        omega_array,
                        force_use_openmp=force_use_openmp,
                        force_no_openmp=force_no_openmp,
                    )

                    # Multiply by source spectrum for each frequency
                    source_mag = source.spectrum_vectorized(omega_array)
                    spectrum = (
                        G_omega * source_mag[:, np.newaxis, np.newaxis]
                    )  # shape (N, 3, 3)

                    return spectrum[0] if squeeze_output else spectrum

                return fortran_callback

            except Exception as e:
                logger.warning(
                    "Failed to initialize Fortran backend: {e}. Falling back to Python.",
                    e=e,
                )
        elif dim == 3 and material_type == MATERIAL_TYPE_ELASTIC:
            # For 3D elastic, use the elastic_wrapper
            rho = material_params["rho"]
            lam = material_params["lam"]
            mu = material_params["mu"]

            def fortran_callback(omega_array: np.ndarray) -> np.ndarray:
                """
                Vectorized evaluation using Fortran backend for 3D elastic.

                Parameters
                ----------
                omega_array : np.ndarray
                    Array of angular frequencies

                Returns
                -------
                spectrum : np.ndarray
                    Shape (N, 6, 6) array of Green's function spectrum times source spectrum
                """
                if np.isscalar(omega_array):
                    omega_array = np.array([omega_array])
                    squeeze_output = True
                else:
                    squeeze_output = False

                G_omega = elastic_wrapper.greens_mixed_force_vectorized(
                    x,
                    omega_array,
                    rho,
                    lam,
                    mu,
                    0,
                    1,
                    0,
                    0,
                    0,  # dummy values for nu, J, ... due to API
                    force_use_openmp=force_use_openmp,
                    force_no_openmp=force_no_openmp,
                )  # shape (n_omega, 6, 6)

                # Get source spectrum
                source_mag = source.spectrum_vectorized(omega_array)
                # Multiply by source magnitude: G * source_magnitude
                spectrum = (
                    G_omega * source_mag[:, np.newaxis, np.newaxis]
                )  # shape (N, 6, 6)

                return spectrum[0] if squeeze_output else spectrum

            return fortran_callback
        elif dim == 3 and material_type == MATERIAL_TYPE_COSSERAT:
            # For 3D Cosserat, use the cosserat_wrapper
            rho = material_params["rho"]
            lam = material_params["lam"]
            mu = material_params["mu"]
            nu = material_params["nu"]
            J = material_params["J"]
            lam_c = material_params["lam_c"]
            mu_c = material_params["mu_c"]
            nu_c = material_params["nu_c"]

            def fortran_callback(omega_array: np.ndarray) -> np.ndarray:
                """
                Vectorized evaluation using Fortran backend for 3D Cosserat.

                Parameters
                ----------
                omega_array : np.ndarray
                    Array of angular frequencies

                Returns
                -------
                spectrum : np.ndarray
                    Shape (N, 6, 6) array of Green's function spectrum times source spectrum
                """
                if np.isscalar(omega_array):
                    omega_array = np.array([omega_array])
                    squeeze_output = True
                else:
                    squeeze_output = False

                G_omega = cosserat_wrapper.greens_mixed_force_vectorized(
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
                    force_use_openmp=force_use_openmp,
                    force_no_openmp=force_no_openmp,
                )  # shape (n_omega, 6, 6)

                # Get source spectrum
                source_mag = source.spectrum_vectorized(omega_array)
                # Multiply by source magnitude: G * source_magnitude
                spectrum = (
                    G_omega * source_mag[:, np.newaxis, np.newaxis]
                )  # shape (N, 6, 6)

                return spectrum[0] if squeeze_output else spectrum

            return fortran_callback

        else:
            err = f"Combination of dimension {dim} and material type {material_type} is not supported for Fortran backend."
            logger.error(err)
            raise ValueError(err)

    # Fall back to Python backend (need to revalidate if Fortran was requested but failed)
    _validate_dimension_backend_material_combo(dim, BACKEND_PYTHON, material_type)

    if dim == 2 and material_type == MATERIAL_TYPE_COSSERAT:
        integrator_python = Integrator(
            rho=material_params["rho"],
            lam=material_params["lam"],
            mu=material_params["mu"],
            nu=material_params["nu"],
            J=material_params["J"],
            lam_c=material_params["lam_c"],
            mu_c=material_params["mu_c"],
            nu_c=material_params["nu_c"],
            digits_precision=digits_precision,
        )

        def python_callback_2d(omega: float) -> np.ndarray:
            """
            Scalar evaluation using Python/mpmath backend in 2D.

            Parameters
            ----------
            omega : float
                Angular frequency (must be scalar)

            Returns
            -------
            spectrum : np.ndarray
                Shape (3, 3) array of Green's function spectrum times source magnitude
            """
            if isinstance(omega, np.ndarray):
                err = (
                    "Python backend only supports scalar omega. "
                    "Use Fortran backend for vectorized evaluation."
                )
                logger.error(err)
                raise ValueError(err)

            # Evaluate Green's function
            G_omega = integrator_python.greens_x_omega(x, omega)

            # Get source spectrum
            source_mag = source.spectrum(omega)

            # Multiply by source magnitude: G * source_magnitude
            return G_omega * source_mag

        return python_callback_2d

    if dim == 3 and material_type == MATERIAL_TYPE_ELASTIC:
        # unpack material parameters
        rho = material_params["rho"]
        lam = material_params["lam"]
        mu = material_params["mu"]

        def python_callback_3d_elastic(omega: float) -> np.ndarray:
            """
            Scalar evaluation using Python/mpmath backend for 3D elastic.

            Parameters
            ----------
            omega : float
                Angular frequency (must be scalar)

            Returns
            -------
            spectrum : np.ndarray
                Shape (6, 6) array of Green's function spectrum times source spectrum
            """
            if isinstance(omega, np.ndarray):
                err = (
                    "Python backend only supports scalar omega. "
                    "Use Fortran backend for vectorized evaluation."
                )
                logger.error(err)
                raise ValueError(err)

            # Evaluate Green's function
            G_omega = elastic.greens_mixed_force(
                x, omega, rho, lam, mu, 0, 1, 0, 0, 0
            )  # dummy values

            # Get source spectrum
            source_mag = source.spectrum(omega)

            # Multiply by source magnitude: G * source_magnitude
            return G_omega * source_mag

        return python_callback_3d_elastic

    if dim == 3 and material_type == MATERIAL_TYPE_COSSERAT:
        # unpack material parameters
        rho = material_params["rho"]
        lam = material_params["lam"]
        mu = material_params["mu"]
        nu = material_params["nu"]
        J = material_params["J"]
        lam_c = material_params["lam_c"]
        mu_c = material_params["mu_c"]
        nu_c = material_params["nu_c"]
        material_type = material_params.get("material_type", "cosserat")

        def python_callback_3d(omega: float) -> np.ndarray:
            """
            Scalar evaluation using Python/mpmath backend for 3D.

            Parameters
            ----------
            omega : float
                Angular frequency (must be scalar)

            Returns
            -------
            spectrum : np.ndarray
                Shape (6, 6) array of Green's function spectrum times source spectrum
            """
            # TODO: check if the python impl for 3d works with vectors
            if isinstance(omega, np.ndarray):
                err = (
                    "Python backend only supports scalar omega. "
                    "Use Fortran backend for vectorized evaluation."
                )
                logger.error(err)
                raise ValueError(err)

            # Evaluate Green's function
            if material_type == "elastic":
                G_omega = elastic.greens_mixed_force(
                    x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
                )
            else:
                G_omega = cosserat.greens_mixed_force(
                    x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
                )

            # Get source spectrum
            source_mag = source.spectrum(omega)

            # Multiply by source magnitude: G * source_magnitude
            return G_omega * source_mag

        return python_callback_3d

    err = f"Combination of dimension {dim} and material type {material_type} is not supported for Python backend."
    logger.error(err)
    raise ValueError(err)
