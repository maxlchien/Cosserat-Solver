from mpmath import mp

from scipy.special import hankel1

class DispersionHelper:
    """
    Helper class for dispersion relations.
    Uses mpmath for high precision complex arithmetic.
    """

    def __init__(self, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, *, digits_precision=30):
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

    # --- dispersion coefficients ---
    def _dispersion_A(self, r):
        r'''
            The coefficient A in the dispersion relation Ac_\pm^2 + Bc_\pm + C = 0
        '''
        r = mp.mpc(r)
        return  mp.mpc(0, 2) * self.nu / mp.sqrt(self.rho * self.J)

    def _dispersion_B(self, r):
        r'''
            The coefficient B in the dispersion relation Ac_\pm^2 + Bc_\pm + C = 0
        '''
        r = mp.mpc(r)
        return (
            r**2 * ((self.mu + self.nu) / self.rho - (self.mu_c + self.nu_c) / self.J)
            - 4 * self.nu / self.J
        )

    def _dispersion_C(self, r):
        r'''
            The coefficient C in the dispersion relation Ac_\pm^2 + Bc_\pm + C = 0
        '''
        r = mp.mpc(r)
        return mp.mpc(0, 2) * self.nu * r**2 / mp.sqrt(self.rho * self.J)

    def _dispersion(self, r, c):
        r'''
            Computes the left hand side of the dispersion relation Ac_\pm^2 + Bc_\pm + C = 0 at c
        '''
        r = mp.mpc(r)
        c = mp.mpc(c)
        A = self._dispersion_A(r)
        B = self._dispersion_B(r)
        C = self._dispersion_C(r)
        return A * c**2 + B * c + C

    def c_pm(self, r, branch):
        r'''
            Solves the dispersion relation Ac_\pm^2 + Bc_\pm + C = 0 for c_\pm, returning the root corresponding to the specified branch.
        '''
        r = mp.mpc(r)
        A = self._dispersion_A(r)
        B = self._dispersion_B(r)
        C = self._dispersion_C(r)
        if mp.re(B) < 0: # B is real
            noncancelling = (-B + mp.sqrt(B**2 - 4 * A * C)) / (2 * A)
            if branch < 0:
                return r ** 2 / noncancelling # vieta's to avoid cancellation
            return noncancelling
        else:
            noncancelling = (-B - mp.sqrt(B**2 - 4 * A * C)) / (2 * A)
            if branch < 0:
                return r ** 2 / noncancelling
            return noncancelling
    
    def c_pm_prime(self, r, branch):
        r'''
            Implicit derivative of c_pm with respect to r.
        '''
        r = mp.mpc(r)
        h = mp.mpf('1e-10')
        c_pm = self.c_pm(r, branch)
        diff_value = (self.mu + self.nu) / self.rho - (self.mu_c + self.nu_c) / self.J
        num = -2 * c_pm * r * diff_value - mp.mpc(0, 4) * self.nu * r / mp.sqrt(self.rho * self.J)
        denom = r ** 2 * diff_value - 4 * self.nu / self.J + mp.mpc(0, 4) * self.nu * c_pm / mp.sqrt(self.rho * self.J)
        return num / denom