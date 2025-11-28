from __future__ import annotations

import numpy as np
import scipy.special
from mpmath import mp

import cosserat_solver.consts as consts
from cosserat_solver.dispersion import DispersionHelper


class Integrator:
    """
    Helper class for Hankel integrals.
    Uses mpmath for high precision complex arithmetic.
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

    def integrate(self, analytic, omega, branch):
        r"""
        Using the residue theorem, integrate the function analytic/denom over the +i0 prescription.
        """
        omega = mp.mpf(omega)
        # TODO: catch omega==0
        poles_and_branches = self.get_poles_and_branches(omega)

        integral = mp.mpc(0)
        for r_pole, pole_branch in poles_and_branches:
            if pole_branch != branch:
                continue
            residue = analytic(r_pole, omega, branch) / self.denom_prime(
                r_pole, omega, branch
            )
            integral += residue

        integral *= mp.mpc(0, 2 * mp.pi)
        return integral

    def _accurate_hankel1(order: int, z: complex) -> complex:
        r"""
        Compute the Hankel function of the first kind of given order to high accuracy for large arguments, combining
            scipy.special.hankel1e and mpmath.
        """
        z = np.complex128(z)
        if np.abs(z) < 10:
            return mp.mpc(scipy.special.hankel1(order, z))
        hankel1e = scipy.special.hankel1e(order, z)
        hankel1e_mp = mp.mpc(hankel1e)
        exp_term = mp.exp(mp.mpc(0, 1) * mp.mpc(z))
        return hankel1e_mp * exp_term

    def integral_3_0(self, normx, omega, branch):
        r"""
        Compute the integral
            \frac{1}{8\pi}\int_{-\infty}^\infty \frac{1}{denom(r, omega, branch)} \frac{r^3}{r^2+\abs{c_\pm}^2}H_0^{(1)}(r\norm{x}) dr
        using the residue theorem. H_0^{(1)} is the Hankel function of the first kind of order 0.
        """

        def analytic(r, _, branch):
            c_pm = self.dispersion_helper.c_pm(r, branch)
            hankel = Integrator._accurate_hankel1(0, r * normx)
            return (r**3 / (r**2 - c_pm**2)) * hankel

        return self.integrate(analytic, omega, branch) / mp.mpc(8 * mp.pi)

    def integral_3_2(self, normx, omega, branch):
        r"""
        Compute the integral
            \frac{1}{8\pi}\int_{-\infty}^\infty \frac{1}{denom(r, omega, branch)} \frac{r^3}{r^2+\abs{c_\pm}^2}H_2^{(1)}(r\norm{x}) dr
        using the residue theorem. H_2^{(1)} is the Hankel function of the first kind of order 2.
        """

        def analytic(r, _, branch):
            c_pm = self.dispersion_helper.c_pm(r, branch)
            hankel = Integrator._accurate_hankel1(2, r * normx)
            return (r**3 / (r**2 - c_pm**2)) * hankel

        return self.integrate(analytic, omega, branch) / mp.mpc(8 * mp.pi)

    def integral_2_1(self, normx, omega, branch):
        r"""
        Compute the integral
            \frac{1}{8\pi}\sqrt{\frac{\rho}{j}}\int_{-\infty}^\infty \frac{1}{denom(r, omega, branch)} \frac{r^2c_\pm}{r^2+\abs{c_\pm}^2}H_1^{(1)}(r\norm{x}) dr
        using the residue theorem. H_1^{(1)} is the Hankel function of the first kind of order 1.
        """

        def analytic(r, _, branch):
            c_pm = self.dispersion_helper.c_pm(r, branch)
            hankel = Integrator._accurate_hankel1(1, r * normx)
            # return (r**2 * c_pm / (r**2 + abs(c_pm) ** 2)) * hankel
            return (r**2 * c_pm / (r**2 - c_pm**2)) * hankel

        return (
            self.integrate(analytic, omega, branch)
            / mp.mpc(8 * mp.pi)
            * mp.sqrt(self.rho / self.J)
        )

    def integral_1_0(self, normx, omega, branch):
        r"""
        Compute the integral
            \frac{1}{8\pi}\frac{\rho}{j}\int_{-\infty}^\infty \frac{1}{denom(r, omega, branch)} \frac{r\abs{c_\pm}^2}{r^2+\abs{c_\pm}^2}H_0^{(1)}(r\norm{x}) dr
        using the residue theorem. H_0^{(1)} is the Hankel function of the first kind of order 0.
        """

        def analytic(r, _, branch):
            c_pm = self.dispersion_helper.c_pm(r, branch)
            hankel = Integrator._accurate_hankel1(0, r * normx)
            # return (r * abs(c_pm) / (r**2 + abs(c_pm) ** 2)) * hankel
            return (-r * c_pm**2 / (r**2 - c_pm**2)) * hankel

        return (
            self.integrate(analytic, omega, branch)
            / mp.mpc(8 * mp.pi)
            * (self.rho / self.J)
        )

    def rotation_matrix(phi):
        r"""
            Compute the in-plane rotation matrix for angle phi.

            This matrix is computed by

            \begin{bmatrix}
                \cos\phi & -\sin\phi & 0\\
                \sin\phi & \cos\phi & 0\\
                0 & 0 & 1
            \end{bmatrix}
        """
        cos_phi = mp.cos(phi)
        sin_phi = mp.sin(phi)
        return mp.matrix(
            [
                [cos_phi, -sin_phi, mp.mpf(0)],
                [sin_phi, cos_phi, mp.mpf(0)],
                [mp.mpf(0), mp.mpf(0), mp.mpf(1)],
            ]
        )

    def greens_x_omega_P(self, x, omega):
        """
        Compute the Green's function G(x, omega) for the P branch.
        """
        normx = mp.norm(x)
        phi = mp.atan2(x[1], x[0])

        c_P = mp.sqrt((self.lam + 2 * self.mu) / self.rho)
        hankel1 = Integrator._accurate_hankel1

        unrotated = mp.matrix(3, 3)
        unrotated[0, 0] = hankel1(0, omega * normx / c_P) - hankel1(
            2, omega * normx / c_P
        )
        unrotated[1, 1] = hankel1(0, omega * normx / c_P) + hankel1(
            2, omega * normx / c_P
        )
        unrotated *= mp.mpc(0, 1) / 8 / (self.lam + 2 * self.mu)

        R = Integrator.rotation_matrix(phi)
        G_mp = R * unrotated * R.T
        # convert to np.ndarray complex floats
        return np.array(G_mp.tolist(), dtype=np.complex128)

    def greens_x_omega_plus(self, x, omega):
        """
        Compute the Green's function G(x, omega) for the + branch.

        Intermediate computation is performed using mpmath precision, but result is converted to np.ndarray complex floats.

        Arguments:
            x: np.ndarray
                2D position vector where to evaluate the Green's function.
            omega: float
                Angular frequency.
        """
        normx = mp.norm(x)
        phi = mp.atan2(x[1], x[0])

        unrotated = mp.matrix(3, 3)
        unrotated[0, 0] = self.integral_3_0(
            normx, omega, consts.PLUS_BRANCH
        ) + self.integral_3_2(normx, omega, consts.PLUS_BRANCH)
        unrotated[1, 1] = self.integral_3_0(
            normx, omega, consts.PLUS_BRANCH
        ) - self.integral_3_2(normx, omega, consts.PLUS_BRANCH)
        unrotated[1, 2] = mp.mpc(0, 1) * self.integral_2_1(
            normx, omega, consts.PLUS_BRANCH
        )
        unrotated[2, 1] = -mp.mpc(0, 1) * self.integral_2_1(
            normx, omega, consts.PLUS_BRANCH
        )
        unrotated[2, 2] = self.integral_1_0(normx, omega, consts.PLUS_BRANCH)

        R = Integrator.rotation_matrix(phi)
        G_mp = R * unrotated * R.T
        # convert to np.ndarray complex floats
        return np.array(G_mp.tolist(), dtype=np.complex128)

    def greens_x_omega_minus(self, x, omega):
        """
        Compute the Green's function G(x, omega) for the - branch.

        Intermediate computation is performed using mpmath precision, but result is converted to np.ndarray complex floats.
        """
        normx = mp.norm(x)
        phi = mp.atan2(x[1], x[0])

        unrotated = mp.matrix(3, 3)
        unrotated[0, 0] = self.integral_3_0(
            normx, omega, -consts.PLUS_BRANCH
        ) + self.integral_3_2(normx, omega, -consts.PLUS_BRANCH)
        unrotated[1, 1] = self.integral_3_0(
            normx, omega, -consts.PLUS_BRANCH
        ) - self.integral_3_2(normx, omega, -consts.PLUS_BRANCH)
        unrotated[1, 2] = mp.mpc(0, 1) * self.integral_2_1(
            normx, omega, -consts.PLUS_BRANCH
        )
        unrotated[2, 1] = -mp.mpc(0, 1) * self.integral_2_1(
            normx, omega, -consts.PLUS_BRANCH
        )
        unrotated[2, 2] = self.integral_1_0(normx, omega, -consts.PLUS_BRANCH)

        R = Integrator.rotation_matrix(phi)
        G_mp = R * unrotated * R.T
        # convert to np.ndarray complex floats
        return np.array(G_mp.tolist(), dtype=np.complex128)

    def greens_x_omega(self, x, omega):
        """
        Compute the Green's function G(x, omega) for both branches.

        Intermediate computation is performed using mpmath precision, but result is converted to np.ndarray complex floats.

        """

        if omega == 0:
            return np.zeros((3, 3), dtype=np.complex128)

        G_P = self.greens_x_omega_P(x, omega)
        G_plus = self.greens_x_omega_plus(x, omega)
        G_minus = self.greens_x_omega_minus(x, omega)

        return G_P + G_plus + G_minus
