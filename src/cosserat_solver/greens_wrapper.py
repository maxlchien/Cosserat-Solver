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
    from cosserat_solver._integrator_core_wrapper import HAS_FORTRAN, IntegratorFortran

    FORTRAN_AVAILABLE = HAS_FORTRAN
except ImportError:
    FORTRAN_AVAILABLE = False
    if TYPE_CHECKING:
        from cosserat_solver._integrator_core_wrapper import IntegratorFortran
    warnings.warn("Fortran backend not available, using Python fallback", stacklevel=2)

import cosserat_solver.cosserat_3d as cosserat_3d
import cosserat_solver.elastic_3d as elastic_3d
from cosserat_solver import consts
from cosserat_solver.integrator import Integrator
from cosserat_solver.source import SourceSpectrum


def _validate_material_params(material_params: dict) -> None:
    """Validate that all required material parameters are present."""
    required_keys = ["rho", "lam", "mu", "nu", "J", "lam_c", "mu_c", "nu_c"]
    missing = [key for key in required_keys if key not in material_params]
    if missing:
        err = f"Missing material parameters: {missing}"
        raise ValueError(err)


def _validate_dimension_python(dim: int) -> None:
    """Validate that dimension is supported for Python evaluation."""
    if dim not in (2, 3):
        err = f"Invalid dimension {dim}. Python backend supports 2D and 3D problems."
        raise ValueError(err)


def _validate_dimension_fortran(dim: int) -> None:
    """Validate that dimension is supported for Fortran evaluation."""
    if dim != 2:
        err = f"Invalid dimension {dim}. Fortran backend currently only supports 2D problems."
        raise ValueError(err)


def evaluate_greens_fortran(
    x: np.ndarray,
    dim: int,
    omega_array: np.ndarray,
    material_params: dict,
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

    Returns
    -------
    greens : np.ndarray
        Complex array of shape (len(omega_array), 3, 3)
        Green's function evaluated at each omega
    """
    if not FORTRAN_AVAILABLE:
        err = "Fortran backend not available."
        raise RuntimeError(err)

    _validate_material_params(material_params)
    _validate_dimension_fortran(dim)

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

    # Ensure x is a list/tuple of length dim, and of type float
    x_nd = np.asarray(x, dtype=float)
    if x_nd.shape != (dim,):
        err = f"Spatial location x must have shape ({dim},) for dimension {dim}."
        raise ValueError(err)

    # Evaluate for each omega
    n_omega = len(omega_array)
    if dim == 2:
        greens = np.zeros((n_omega, 3, 3), dtype=np.complex128)
    else:
        err = f"Invalid dimension {dim}. Fortran backend currently only supports 2D problems."
        raise ValueError(err)

    for i, omega in enumerate(omega_array):
        # Convert omega to complex
        omega_complex = complex(omega)
        # Get Green's function as nested tuple
        G_tuple = integrator.greens_x_omega(x_nd.tolist(), omega_complex)
        # Convert to numpy array
        greens[i] = np.array(G_tuple, dtype=np.complex128)

    return greens


def evaluate_greens_python(
    x: np.ndarray,
    dim: int,
    omega: float,
    material_params: dict,
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
    _validate_dimension_python(dim)

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
    raise NotImplementedError(err)


def get_greens_callback(
    x: np.ndarray,
    dim: int,
    material_params: dict,
    source: SourceSpectrum,
    use_fortran: bool = True,
    digits_precision: int = consts.COMPUTE_PRECISION,
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
    use_fortran : bool, default=True
        Whether to use Fortran backend if available. Falls back to Python if not.
    digits_precision : int
        Number of digits for mpmath precision (Python backend only)

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

    # Try to use Fortran backend
    if use_fortran and FORTRAN_AVAILABLE:
        _validate_dimension_fortran(dim)
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

                n_omega = len(omega_array)
                spectrum = np.zeros((n_omega, 3, 3), dtype=np.complex128)

                for i, omega in enumerate(omega_array):
                    # Evaluate Green's function
                    G_tuple = integrator_fortran.greens_x_omega(x_2d, complex(omega))
                    G = np.array(G_tuple, dtype=np.complex128)  # shape (3, 3)

                    # Get source spectrum
                    source_mag = source.spectrum(float(omega))

                    # Multiply by source magnitude: G * source_magnitude
                    spectrum[i] = G * source_mag

                return spectrum[0] if squeeze_output else spectrum

            return fortran_callback

        except Exception as e:
            warnings.warn(
                f"Failed to initialize Fortran backend: {e}. Falling back to Python.",
                stacklevel=2,
            )

    # Fall back to Python backend
    _validate_dimension_python(dim)

    if dim == 2:
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
                raise ValueError(err)

            # Evaluate Green's function
            G_omega = integrator_python.greens_x_omega(x, omega)

            # Get source spectrum
            source_mag = source.spectrum(omega)

            # Multiply by source magnitude: G * source_magnitude
            return G_omega * source_mag

        return python_callback_2d

    if dim == 3:
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
                raise ValueError(err)

            # Evaluate Green's function
            if material_type == "elastic":
                G_omega = elastic_3d.greens_mixed_force(
                    x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
                )
            else:
                G_omega = cosserat_3d.greens_mixed_force(
                    x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c
                )

            # Get source spectrum
            source_mag = source.spectrum(omega)

            # Multiply by source magnitude: G * source_magnitude
            return G_omega * source_mag

        return python_callback_3d

    err = f"Invalid dimension {dim}. Python backend currently only supports 2D and 3D problems."
    raise NotImplementedError(err)
