from __future__ import annotations

import itertools

import pytest
from mpmath import mp

import cosserat_solver.consts as consts
from cosserat_solver.integrator import Integrator

mp.dps = consts.TEST_PRECISION

# TODO: merge the test suite sets
# --- Predefined material parameter sets ---
MATERIAL_PARAMETER_SETS = [
    {
        "rho": mp.mpf("1e3"),
        "lam": mp.mpf("1e5"),
        "mu": mp.mpf("1e5"),
        "nu": mp.mpf("1e4"),
        "J": mp.mpf("1"),
        "lam_c": mp.mpf("1e5"),
        "mu_c": mp.mpf("1e5"),
        "nu_c": mp.mpf("1e4"),
    },
    {
        "rho": mp.mpf("1e4"),
        "lam": mp.mpf("1e6"),
        "mu": mp.mpf("5e6"),
        "nu": mp.mpf("1e5"),
        "J": mp.mpf("1e2"),
        "lam_c": mp.mpf("2e6"),
        "mu_c": mp.mpf("3e6"),
        "nu_c": mp.mpf("5e5"),
    },
    {
        "rho": mp.mpf("1e5"),
        "lam": mp.mpf("1e7"),
        "mu": mp.mpf("1e7"),
        "nu": mp.mpf("1e6"),
        "J": mp.mpf("1e6"),
        "lam_c": mp.mpf("1e7"),
        "mu_c": mp.mpf("1e7"),
        "nu_c": mp.mpf("1e6"),
    },
    {
        "rho": mp.mpf("1e6"),
        "lam": mp.mpf("1e8"),
        "mu": mp.mpf("5e7"),
        "nu": mp.mpf("1e7"),
        "J": mp.mpf("1e3"),
        "lam_c": mp.mpf("1e8"),
        "mu_c": mp.mpf("1e8"),
        "nu_c": mp.mpf("1e7"),
    },
    {
        "rho": mp.mpf("1e8"),
        "lam": mp.mpf("5e5"),
        "mu": mp.mpf("2e8"),
        "nu": mp.mpf("3e6"),
        "J": mp.mpf("1e4"),
        "lam_c": mp.mpf("4e7"),
        "mu_c": mp.mpf("6e7"),
        "nu_c": mp.mpf("8e6"),
    },
]

# --- Define k and omega ranges ---
K_VALUES = [
    mp.mpf("1e-1"),
    mp.mpf("1.0"),
    mp.mpf("10.0"),
    mp.mpf("100.0"),
    mp.mpf("1000.0"),
]
OMEGA_VALUES = [mp.mpf("1e1"), mp.mpf("1e2"), mp.mpf("1e3"), mp.mpf("1e4")]


# --- Fixture for material parameters ---
@pytest.fixture(params=MATERIAL_PARAMETER_SETS)
def material_parameters(request):
    """Fixture providing predefined sets of material parameters."""
    return request.param


# --- Fixture for (k, omega) combinations ---
@pytest.fixture(
    params=[
        {"k": k, "omega": omega}
        for k, omega in itertools.product(K_VALUES, OMEGA_VALUES)
    ]
)
def wave_parameters(request):
    """Fixture providing combinations of wavenumber and frequency."""
    return request.param


@pytest.fixture(params=[consts.PLUS_BRANCH, -consts.PLUS_BRANCH])
def branch(request):
    """Fixture providing the branch of the dispersion relation."""
    return request.param


def test_denom_prime(material_parameters, wave_parameters, branch):
    """Test that the derivative of the denominator function is computed correctly."""
    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = wave_parameters["k"]
    omega = wave_parameters["omega"]

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    # Numerical derivative using central difference
    h = mp.mpf("1e-6")
    denom_plus = integrator.denom(k + h, omega, branch)
    denom_minus = integrator.denom(k - h, omega, branch)
    numerical_derivative = (denom_plus - denom_minus) / (2 * h)

    analytic_derivative = integrator.denom_prime(k, omega, branch)

    assert mp.almosteq(numerical_derivative, analytic_derivative, abs_eps=1e-6), (
        f"Denominator derivative mismatch for params: {params}, k: {k}, omega: {omega}, branch: {branch}"
    )


def test_alternate_pole_repr(material_parameters, wave_parameters):
    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = wave_parameters["omega"]

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    r2_poles_and_branches = integrator._get_r2_poles_and_branches(omega)

    for r2, _ in r2_poles_and_branches:
        c_p = integrator.dispersion_helper.c_pm(mp.sqrt(r2), consts.PLUS_BRANCH)
        c_m = integrator.dispersion_helper.c_pm(mp.sqrt(r2), -consts.PLUS_BRANCH)
        rhs = mp.sqrt(J / rho) * (rho * omega**2 - (mu + nu) * r2) / (mp.mpc(0, 2) * nu)
        assert mp.almosteq(c_p, rhs, abs_eps=1e-6) or mp.almosteq(
            c_m, rhs, abs_eps=1e-6
        ), (
            f"Alternate pole representation mismatch for params: {params}, r2: {r2}, omega: {omega}"
        )


def test_denom_zeros(material_parameters, wave_parameters):
    """Test that the denominator function evaluates to zero at the computed poles."""
    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = wave_parameters["omega"]

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    r2_poles_and_branches = integrator._get_r2_poles_and_branches(omega)
    # # this is a bit tricky since each r^2 pole corresponds to a c_\pm branch, but we "forget" which one
    # d1_unswitched, d2_unswitched = integrator.denom(mp.sqrt(r21), omega, branch), integrator.denom(mp.sqrt(r22), omega, -branch)
    # d1_switched, d2_switched = integrator.denom(mp.sqrt(r21), omega, -branch), integrator.denom(mp.sqrt(r22), omega, branch)

    # assert (mp.almosteq(d1_unswitched, 0.0, abs_eps=1e-6) and mp.almosteq(d2_unswitched, 0.0, abs_eps=1e-6)) or \
    #           (mp.almosteq(d1_switched, 0.0, abs_eps=1e-6) and mp.almosteq(d2_switched, 0.0, abs_eps=1e-6)), f"Denominator not zero at poles for params: {params}, omega: {omega}, branch: {branch}"
    for r2, b in r2_poles_and_branches:
        r = mp.sqrt(r2)
        denom_value = integrator.denom(r, omega, b)
        assert mp.almosteq(denom_value, 0.0, abs_eps=1e-6), (
            f"Denominator not zero at pole for params: {params}, r: {r}, omega: {omega}"
        )


def test_combined_equation_1(material_parameters, wave_parameters):
    """Test that the first expression eliminating c_pm is satisfied at the poles."""
    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = wave_parameters["omega"]

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    r2_poles_and_branches = integrator._get_r2_poles_and_branches(omega)
    for r2, _ in r2_poles_and_branches:
        r = mp.sqrt(r2)
        # \sqrt{\frac{j}{\rho}} \frac{1}{\rho}\paren{\rho\omega^2-(\mu+\nu)r^2}^2+\sqrt{\frac{j}{\rho}}\paren{r^2\paren{\frac{\mu+\nu}{\rho}-\frac{\mu_c+\nu_c}{j}}-\frac{4\nu}{j}}\paren{\rho\omega^2-(\mu+\nu)r^2}-\frac{4\nu^2}{\sqrt{\rho j} }r^2=0
        val = (
            mp.sqrt(J / rho) * (1 / rho) * (rho * omega**2 - (mu + nu) * r2) ** 2
            + mp.sqrt(J / rho)
            * (r2 * ((mu + nu) / rho - (mu_c + nu_c) / J) - (4 * nu) / J)
            * (rho * omega**2 - (mu + nu) * r2)
            - (4 * nu**2) / mp.sqrt(rho * J) * r2
        )
        assert mp.almosteq(val, 0.0, abs_eps=1e-6), (
            f"Combined equation not satisfied for params: {params}, r: {r}, omega: {omega}"
        )


def test_combined_equation_2(material_parameters, wave_parameters):
    """Test that the second expression eliminating c_pm is satisfied at the poles."""
    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = wave_parameters["omega"]

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    r2_poles_and_branches = integrator._get_r2_poles_and_branches(omega)
    for r2, _ in r2_poles_and_branches:
        r = mp.sqrt(r2)
        # r^4 \paren{\frac{(\mu+\nu)(\mu_c+\nu_c)}{\sqrt{\rho j} }}+r^2\paren{-\sqrt{\rho j}\paren{\frac{\mu+\nu}{\rho}+\frac{\mu_c+\nu_c}{j}}\omega^2+\frac{4\nu \mu}{\sqrt{\rho j} } } + \sqrt{\rho j} \omega^4 - \sqrt{\frac{\rho}{j}}4\nu\omega^2  =0
        val = (
            r2**2 * ((mu + nu) * (mu_c + nu_c) / mp.sqrt(rho * J))
            + r2
            * (
                -mp.sqrt(rho * J) * ((mu + nu) / rho + (mu_c + nu_c) / J) * omega**2
                + (4 * nu * mu) / mp.sqrt(rho * J)
            )
            + mp.sqrt(rho * J) * omega**4
            - mp.sqrt(rho / J) * 4 * nu * omega**2
        )
        assert mp.almosteq(val, 0.0, abs_eps=1e-6), (
            f"Combined equation not satisfied for params: {params}, r: {r}, omega: {omega}"
        )


def test_dispersion_r2(material_parameters, wave_parameters):
    """Test that the dispersion relation is satisfied at the poles."""
    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = wave_parameters["omega"]

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    r2_poles_and_branches = integrator._get_r2_poles_and_branches(omega)
    for r2, b in r2_poles_and_branches:
        r = mp.sqrt(r2)
        c_pm = integrator.dispersion_helper.c_pm(r, b)
        dispersion_value = integrator.dispersion_helper._dispersion(r, c_pm)
        assert mp.almosteq(dispersion_value, 0.0, abs_eps=1e-6), (
            f"Dispersion relation not satisfied for params: {params}, r: {r}, omega: {omega}"
        )


# a lot of complex values to test
R2_TEST_VALUES = [
    mp.mpc("1e-2", "1e-2"),
    mp.mpc("1e0", "1e0"),
    mp.mpc("1e2", "1e2"),
    mp.mpc("1e4", "1e4"),
    mp.mpc("1e6", "1e6"),
    mp.mpc("1e8", "1e8"),
    mp.mpc("1e1", "-1e1"),
    mp.mpc("1e3", "-1e3"),
    mp.mpc("1e5", "-1e5"),
    mp.mpc("1e7", "-1e7"),
    mp.mpc("1e9", "-1e9"),
    mp.mpc("-1e1", "1e1"),
    mp.mpc("-1e3", "1e3"),
    mp.mpc("-1e5", "1e5"),
    mp.mpc("-1e7", "1e7"),
    mp.mpc("-1e9", "1e9"),
    mp.mpc("-1e2", "-1e2"),
    mp.mpc("-1e4", "-1e4"),
    mp.mpc("-1e6", "-1e6"),
    mp.mpc("-1e8", "-1e8"),
]


@pytest.fixture(params=R2_TEST_VALUES)
def r2_value(request):
    return request.param


def test_pole_selection(r2_value, wave_parameters):
    """Test that the pole selection always produces poles in the upper half plane, or on the side of the real axis corresponding to
    the sign of omega"""
    omega = wave_parameters["omega"]

    r = Integrator._pick_pole(r2_value, omega)
    # check that the pole is in the upper half plane, or on the correct side of the real axis
    in_uhp = mp.im(r) > 0 or (
        mp.almosteq(mp.im(r), 0.0, abs_eps=1e-6) and (mp.re(r) * mp.re(omega) >= 0)
    )
    assert in_uhp, (
        f"Pole selection failed for r^2: {r2_value}, omega: {omega}, selected r: {r}"
    )


def test_pole_locations(material_parameters, wave_parameters):
    """Test that the computed poles have non-negative imaginary parts, or lie on the correct side of the real axis."""
    params = material_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    omega = wave_parameters["omega"]

    integrator = Integrator(
        rho=rho,
        lam=lam,
        mu=mu,
        nu=nu,
        J=J,
        lam_c=lam_c,
        mu_c=mu_c,
        nu_c=nu_c,
        digits_precision=consts.TEST_PRECISION,
    )

    r2_poles_and_branches = integrator._get_r2_poles_and_branches(omega)
    for r2, _ in r2_poles_and_branches:
        r = mp.sqrt(r2)
        in_uhp = mp.im(r) > 0 or (
            mp.almosteq(mp.im(r), 0.0, abs_eps=1e-6) and (mp.re(r) * mp.re(omega) >= 0)
        )
        assert in_uhp, (
            f"Pole location incorrect for params: {params}, r^2: {r2}, omega: {omega}, selected r: {r}"
        )
