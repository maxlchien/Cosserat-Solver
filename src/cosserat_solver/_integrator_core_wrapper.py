"""
Python wrapper for integrator_core Fortran extension.
"""

from __future__ import annotations

try:
    from cosserat_solver import integrator_core

    HAS_FORTRAN = True
except ImportError:
    HAS_FORTRAN = False


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
