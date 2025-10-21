from __future__ import annotations

from mpmath import mp

import cosserat_solver.consts as consts
from cosserat_solver.dispersion import DispersionHelper


class Integrator:
    """
    Helper class for Hankel integrals.
    Supports a high-precision mode using mpmath for complex arithmetic.
    """

    def __init__(self, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, *, digits_precision=50):
        # --- parameters ---
        self.rho = mp.mpf(rho)
        self.lam = mp.mpf(lam)
        self.mu = mp.mpf(mu)
        self.nu = mp.mpf(nu)
        self.J = mp.mpf(J)
        self.lam_c = mp.mpf(lam_c)
        self.mu_c = mp.mpf(mu_c)
        self.nu_c = mp.mpf(nu_c)

        mp.dps = digits_precision

        self.dispersion_helper = DispersionHelper(
            rho, lam, mu, nu, J, lam_c, mu_c, nu_c, digits_precision=digits_precision
        )

    def denom(self, r, omega, branch):
        r"""
        Denominator function for Hankel integrals.
        """
        r = mp.mpc(r)
        omega = mp.mpf(omega)
        c_pm = self.dispersion_helper.c_pm(r, branch)
        return (
            (self.mu + self.nu) * r**2
            - self.rho * omega**2
            + mp.mpc(0, 2) * self.nu * mp.sqrt(self.rho / self.J) * c_pm
        )

    def denom_prime(self, r, omega, branch):
        r"""
        Derivative of the denominator function for Hankel integrals.
        """
        r = mp.mpc(r)
        omega = mp.mpf(omega)
        return 2 * (self.mu + self.nu) * r + mp.mpc(0, 2) * self.nu * mp.sqrt(
            self.rho / self.J
        ) * self.dispersion_helper.c_pm_prime(r, branch)

    def _get_r2_poles_and_branches(self, omega):
        r"""
        Computes the values of r^2, where r are the poles of the integrand in the Hankel integrals. Also return the corresponding
        branch of c_\pm for each pole.

        Returns:
            List of tuples (r^2, branch)
        """
        omega = mp.mpf(omega)
        a_4 = (self.mu + self.nu) * (self.mu_c + self.nu_c) / mp.sqrt(self.rho * self.J)
        a_2 = -mp.sqrt(self.rho * self.J) * (
            (self.mu + self.nu) / self.rho + (self.mu_c + self.nu_c) / self.J
        ) * omega**2 + 4 * self.nu * self.mu / mp.sqrt(self.rho * self.J)
        a_0 = (
            mp.sqrt(self.rho * self.J) * omega**4
            - mp.sqrt(self.rho * self.J) * (4 * self.nu / self.J) * omega**2
        )

        # compute one using quadratic formula, then the other using Vieta's formula to minimize numerical error
        if a_2 < 0:
            pole1 = (-a_2 + mp.sqrt(a_2**2 - 4 * a_4 * a_0)) / (2 * a_4)
            pole2 = a_0 / (a_4 * pole1)
            poles = [pole1, pole2]
        else:
            pole2 = (-a_2 - mp.sqrt(a_2**2 - 4 * a_4 * a_0)) / (2 * a_4)
            pole1 = a_0 / (a_4 * pole2)
            poles = [pole1, pole2]

        # check which branch each pole corresponds to
        # later this should be done analytically
        rtn = []
        for pole in poles:
            if mp.almosteq(
                self.denom(mp.sqrt(pole), omega, consts.PLUS_BRANCH), 0.0, abs_eps=1e-6
            ):
                rtn.append((pole, consts.PLUS_BRANCH))
            elif mp.almosteq(
                self.denom(mp.sqrt(pole), omega, -consts.PLUS_BRANCH), 0.0, abs_eps=1e-6
            ):
                rtn.append((pole, -consts.PLUS_BRANCH))
            else:
                msg = f"Could not determine branch for pole r^2={pole} at omega={omega}"
                raise ValueError(msg)
        return rtn

    def _pick_pole(r2, omega):
        r"""
        Given a pole r^2, return the value of r captured by the pole
        """
        r2 = mp.mpc(r2)
        omega = mp.mpf(omega)
        if mp.almosteq(mp.im(r2), 0.0, abs_eps=1e-15) and mp.re(r2) > 0:
            if mp.re(omega) >= 0:
                return mp.sqrt(mp.re(r2))
            return -mp.sqrt(mp.re(r2))
        r = mp.sqrt(r2)
        if mp.im(r) > 0:
            return r
        return -r

    def get_poles_and_branches(self, omega):
        r"""
        Computes the captured poles of the integrand in the Hankel integrals and their corresponding branches.

        Returns:
            List of tuples (r, branch)
        """
        omega = mp.mpf(omega)
        r2_poles_and_branches = self._get_r2_poles_and_branches(omega)
        rtn = []
        for r2, branch in r2_poles_and_branches:
            r = Integrator._pick_pole(r2, omega)
            rtn.append((r, branch))
        return rtn
