from __future__ import annotations

try:
    from cosserat_solver import dispersion_core

    HAS_FORTRAN = True
except ImportError:
    HAS_FORTRAN = False

import numpy as np


class DispersionHelperFortran:
    def __init__(self, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, digits_precision=30):  # noqa: ARG002
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
        dispersion_core.init_dispersion(
            self.rho,
            self.lam,
            self.mu,
            self.nu,
            self.J,
            self.lam_c,
            self.mu_c,
            self.nu_c,
        )

    def c_pm(self, r: complex, branch: int) -> complex:
        real = float(np.real(r))
        imag = float(np.imag(r))

        result_real, result_imag = dispersion_core.c_pm(real, imag, branch)
        return complex(result_real, result_imag)

    def c_pm_prime(self, r: complex, branch: int) -> complex:
        real = float(np.real(r))
        imag = float(np.imag(r))

        result_real, result_imag = dispersion_core.c_pm_prime(real, imag, branch)
        return complex(result_real, result_imag)

    def _dispersion_A(self, r: complex) -> complex:
        real = float(np.real(r))
        imag = float(np.imag(r))

        result_real, result_imag = dispersion_core.dispersion_A(real, imag)
        return complex(result_real, result_imag)

    def _dispersion_B(self, r: complex) -> complex:
        real = float(np.real(r))
        imag = float(np.imag(r))

        result_real, result_imag = dispersion_core.dispersion_B(real, imag)
        return complex(result_real, result_imag)

    def _dispersion_C(self, r: complex) -> complex:
        real = float(np.real(r))
        imag = float(np.imag(r))

        result_real, result_imag = dispersion_core.dispersion_C(real, imag)
        return complex(result_real, result_imag)

    def _dispersion(self, r: complex, c: complex) -> complex:
        r_real = float(np.real(r))
        r_imag = float(np.imag(r))
        c_real = float(np.real(c))
        c_imag = float(np.imag(c))
        result_real, result_imag = dispersion_core.dispersion(
            r_real, r_imag, c_real, c_imag
        )
        return complex(result_real, result_imag)

    def _dispersion_zero(self, r: complex, branch: int) -> complex:
        r_real = float(np.real(r))
        r_imag = float(np.imag(r))
        result_real, result_imag = dispersion_core.dispersion_zero(
            r_real, r_imag, branch
        )
        return complex(result_real, result_imag)
