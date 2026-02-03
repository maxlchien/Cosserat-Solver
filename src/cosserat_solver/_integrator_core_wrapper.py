"""
Python wrapper for integrator_core Fortran extension.
"""

from __future__ import annotations

import typing
from ctypes import Structure, addressof, c_double, c_int32, c_void_p

import numpy as np
from scipy import LowLevelCallable

try:
    from cosserat_solver import integrator_core

    HAS_FORTRAN = True
except ImportError:
    HAS_FORTRAN = False


class IntegrandContext(Structure):
    _fields_: typing.ClassVar[list[tuple[str, type]]] = [
        ("omega_re", c_double),
        ("omega_im", c_double),
        ("normx_re", c_double),
        ("normx_im", c_double),
        ("branch", c_int32),
        ("rho", c_double),
        ("lam", c_double),
        ("mu", c_double),
        ("nu", c_double),
        ("J", c_double),
        ("lam_c", c_double),
        ("mu_c", c_double),
        ("nu_c", c_double),
        ("contour_shift", c_double),
        ("integrand_id", c_int32),
        ("component", c_int32),
    ]


class IntegratorFortran:
    """
    Python-side wrapper for Cosserat Green's function integrator in Fortran.

    All complex quantities are represented as Python complex numbers.
    Branch indices are passed explicitly.
    """

    def __init__(
        self,
        rho,
        lam,
        mu,
        nu,
        J,
        lam_c,
        mu_c,
        nu_c,
    ):
        """
        Store material parameters as instance attributes.

        Material parameters are passed directly to Fortran calls,
        allowing multiple parallel instances.
        """
        if not HAS_FORTRAN:
            err = "Fortran module not available."
            raise RuntimeError(err)

        self.rho = float(rho)
        self.lam = float(lam)
        self.mu = float(mu)
        self.nu = float(nu)
        self.J = float(J)
        self.lam_c = float(lam_c)
        self.mu_c = float(mu_c)
        self.nu_c = float(nu_c)

    # ------------------------------------------------------------
    # Denominator and derivative
    # ------------------------------------------------------------

    def denom(self, r: complex, omega: complex, branch: int) -> complex:
        """
        Evaluate dispersion denominator D(r, omega) on a given branch.
        """
        re, im = integrator_core.denom(
            r.real,
            r.imag,
            omega.real,
            omega.imag,
            int(branch),
            self.rho,
            self.mu,
            self.nu,
            self.J,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    def denom_prime(self, r: complex, omega: complex, branch: int) -> complex:
        """
        Evaluate ∂D/∂r on a given branch.
        """
        re, im = integrator_core.denom_prime(
            r.real,
            r.imag,
            omega.real,
            omega.imag,
            int(branch),
            self.rho,
            self.mu,
            self.nu,
            self.J,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    # ------------------------------------------------------------
    # Poles and branch selection
    # ------------------------------------------------------------

    def get_r2_poles_and_branches(self, omega: complex):
        """
        Return (r2_poles, branches), where

          r2_poles : list[complex]
          branches : list[int]

        Length is typically 1 or 2 depending on ω.
        """
        poles, branches = integrator_core.get_r2_poles_and_branches(
            omega.real,
            omega.imag,
            self.rho,
            self.mu,
            self.nu,
            self.J,
            self.mu_c,
            self.nu_c,
        )

        r2 = [complex(re, im) for (re, im) in poles]
        br = [int(b) for b in branches]

        return r2, br

    def pick_pole(self, r2: complex, omega: complex) -> complex:
        """
        Select the physical square-root pole r from r^2.
        """
        re, im = integrator_core.pick_pole(
            r2.real,
            r2.imag,
            omega.real,
            omega.imag,
        )
        return re + 1j * im

    # ------------------------------------------------------------
    # Integral kernels
    # ------------------------------------------------------------

    def integral_3_0(self, omega: complex, normx: complex, branch: int) -> complex:
        re, im = integrator_core.integral_3_0(
            omega.real,
            omega.imag,
            normx.real,
            normx.imag,
            int(branch),
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    def integral_3_2(self, omega: complex, normx: complex, branch: int) -> complex:
        re, im = integrator_core.integral_3_2(
            omega.real,
            omega.imag,
            normx.real,
            normx.imag,
            int(branch),
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    def integral_2_1(self, omega: complex, normx: complex, branch: int) -> complex:
        re, im = integrator_core.integral_2_1(
            omega.real,
            omega.imag,
            normx.real,
            normx.imag,
            int(branch),
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    def integral_1_0(self, omega: complex, normx: complex, branch: int) -> complex:
        re, im = integrator_core.integral_1_0(
            omega.real,
            omega.imag,
            normx.real,
            normx.imag,
            int(branch),
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    # ------------------------------------------------------------
    # Integrand kernels
    # ------------------------------------------------------------

    def integrand_3_0(
        self, r: complex, omega: complex, normx: complex, branch: int
    ) -> complex:
        re, im = integrator_core.integrand_3_0(
            r.real,
            r.imag,
            omega.real,
            omega.imag,
            normx.real,
            normx.imag,
            int(branch),
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    def integrand_3_2(
        self, r: complex, omega: complex, normx: complex, branch: int
    ) -> complex:
        re, im = integrator_core.integrand_3_2(
            r.real,
            r.imag,
            omega.real,
            omega.imag,
            normx.real,
            normx.imag,
            int(branch),
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    def integrand_2_1(
        self, r: complex, omega: complex, normx: complex, branch: int
    ) -> complex:
        re, im = integrator_core.integrand_2_1(
            r.real,
            r.imag,
            omega.real,
            omega.imag,
            normx.real,
            normx.imag,
            int(branch),
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    def integrand_1_0(
        self, r: complex, omega: complex, normx: complex, branch: int
    ) -> complex:
        re, im = integrator_core.integrand_1_0(
            r.real,
            r.imag,
            omega.real,
            omega.imag,
            normx.real,
            normx.imag,
            int(branch),
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )
        return re + 1j * im

    def lowlevel_integrand(
        self,
        integrand_id: int,
        omega: complex,
        normx: complex,
        branch: int,
        *,
        component: int,
        contour_shift: float = 1e-8,
    ):
        """
        Build a SciPy LowLevelCallable and context for fast quad integration.

        Returns (lowlevel_callable, context). Keep the context alive while integrating.
        """

        ctx = IntegrandContext(
            omega.real,
            omega.imag,
            normx.real,
            normx.imag,
            int(branch),
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
            float(contour_shift),
            int(integrand_id),
            int(component),
        )

        # Get the C function capsule
        capsule = integrator_core.integrand_llc_capsule()

        llc = LowLevelCallable(capsule, c_void_p(addressof(ctx)))
        return llc, ctx

    # ------------------------------------------------------------
    # Green's functions
    # ------------------------------------------------------------

    def greens_x_omega_P(self, x: tuple | list, omega: complex) -> tuple:
        """
        Compute the Green's function G(x, omega) for the P branch.

        Parameters
        ----------
        x : tuple | list
            2D position vector [x1, x2]
        omega : complex
            Angular frequency

        Returns
        -------
        tuple
            3x3 matrix as tuple of tuples, each element is a complex number
        """
        return integrator_core.greens_x_omega_P_c(
            x,
            omega.real,
            omega.imag,
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )

    def greens_x_omega_plus(self, x: tuple | list, omega: complex) -> tuple:
        """
        Compute the Green's function G(x, omega) for the + branch.

        Parameters
        ----------
        x : tuple | list
            2D position vector [x1, x2]
        omega : complex
            Angular frequency

        Returns
        -------
        tuple
            3x3 matrix as tuple of tuples, each element is a complex number
        """
        return integrator_core.greens_x_omega_plus_c(
            x,
            omega.real,
            omega.imag,
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )

    def greens_x_omega_minus(self, x: tuple | list, omega: complex) -> tuple:
        """
        Compute the Green's function G(x, omega) for the - branch.

        Parameters
        ----------
        x : tuple | list
            2D position vector [x1, x2]
        omega : complex
            Angular frequency

        Returns
        -------
        tuple
            3x3 matrix as tuple of tuples, each element is a complex number
        """
        return integrator_core.greens_x_omega_minus_c(
            x,
            omega.real,
            omega.imag,
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )

    def greens_x_omega(self, x: tuple | list, omega: complex) -> tuple:
        """
        Compute the Green's function G(x, omega) for all branches.

        This combines the P, +, and - branch contributions.

        Parameters
        ----------
        x : tuple | list
            2D position vector [x1, x2]
        omega : complex
            Angular frequency

        Returns
        -------
        tuple
            3x3 matrix as tuple of tuples, each element is a complex number
        """
        return integrator_core.greens_x_omega_c(
            x,
            omega.real,
            omega.imag,
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )

    def greens_x_omega_vectorized(
        self,
        x: tuple | list,
        omega: complex | np.ndarray,
        force_use_openmp: bool = False,
        force_no_openmp: bool = False,
    ) -> np.ndarray:
        """
        Compute the Green's function G(x, omega) for all branches.

        Automatically detects whether omega is scalar or array and handles appropriately.
        Scalar inputs are converted to length-1 arrays internally for unified processing.

        Parameters
        ----------
        x : tuple | list
            2D position vector [x1, x2]
        omega : complex | np.ndarray
            Angular frequency (scalar) or array of angular frequencies
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
            omega_array = np.array([complex(omega)], dtype=np.complex128)
            squeeze_output = True
        else:
            # Array case
            omega_array = np.asarray(omega, dtype=np.complex128)
            if omega_array.ndim != 1:
                err = "omega must be scalar or 1D array"
                raise ValueError(err)
            squeeze_output = False

        # Call vectorized Fortran backend
        # Returns list of tuples (one per omega)
        result_list = integrator_core.greens_x_omega_vectorized_c(
            x,
            omega_array,
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
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
            return result_array[0]
        return result_array
