from cosserat_solver.dispersion import DispersionHelper
import pytest
import itertools
import cosserat_solver.consts as consts
from mpmath import mp

mp.dps = consts.TEST_PRECISION

# --- Predefined material parameter sets ---
MATERIAL_PARAMETER_SETS = [
    {
        "rho": mp.mpf('1e3'),
        "lam": mp.mpf('1e5'),
        "mu": mp.mpf('1e5'),
        "nu": mp.mpf('1e4'),
        "J": mp.mpf('1'),
        "lam_c": mp.mpf('1e5'),
        "mu_c": mp.mpf('1e5'),
        "nu_c": mp.mpf('1e4'),
    },
    {
        "rho": mp.mpf('1e4'),
        "lam": mp.mpf('1e6'),
        "mu": mp.mpf('5e6'),
        "nu": mp.mpf('1e5'),
        "J": mp.mpf('1e2'),
        "lam_c": mp.mpf('2e6'),
        "mu_c": mp.mpf('3e6'),
        "nu_c": mp.mpf('5e5'),
    },
    {
        "rho": mp.mpf('1e5'),
        "lam": mp.mpf('1e7'),
        "mu": mp.mpf('1e7'),
        "nu": mp.mpf('1e6'),
        "J": mp.mpf('1e6'),
        "lam_c": mp.mpf('1e7'),
        "mu_c": mp.mpf('1e7'),
        "nu_c": mp.mpf('1e6'),
    },
    {
        "rho": mp.mpf('1e6'),
        "lam": mp.mpf('1e8'),
        "mu": mp.mpf('5e7'),
        "nu": mp.mpf('1e7'),
        "J": mp.mpf('1e3'),
        "lam_c": mp.mpf('1e8'),
        "mu_c": mp.mpf('1e8'),
        "nu_c": mp.mpf('1e7'),
    },
    {
        "rho": mp.mpf('1e8'),
        "lam": mp.mpf('5e5'),
        "mu": mp.mpf('2e8'),
        "nu": mp.mpf('3e6'),
        "J": mp.mpf('1e4'),
        "lam_c": mp.mpf('4e7'),
        "mu_c": mp.mpf('6e7'),
        "nu_c": mp.mpf('8e6'),
    },
]

# --- Define k and omega ranges ---
K_VALUES = [mp.mpf('1e-1'), mp.mpf('1.0'), mp.mpf('10.0'), mp.mpf('100.0'), mp.mpf('1000.0')]
OMEGA_VALUES = [mp.mpf('1e1'), mp.mpf('1e2'), mp.mpf('1e3'), mp.mpf('1e4')]

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

def test_imaginary(material_parameters, wave_parameters, branch):
    """Test that c_pm is entirely imaginary for given parameters."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]

    dispersion_helper = DispersionHelper(rho = rho, lam=lam, mu=mu, nu=nu,
                            J=J, lam_c=lam_c, mu_c=mu_c, nu_c=nu_c, digits_precision=consts.TEST_PRECISION)
    
    c_value = dispersion_helper.c_pm(k, branch)
    assert mp.almosteq(c_value.real, 0.0, abs_eps=1e-6), f"Real part of c_pm is not close to zero for params: {params}, branch: {branch}"

def test_real_B(material_parameters, wave_parameters):
    """Test that dispersion coefficient B is a real number."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]

    dispersion_helper = DispersionHelper(rho = rho, lam=lam, mu=mu, nu=nu,
                            J=J, lam_c=lam_c, mu_c=mu_c, nu_c=nu_c, digits_precision=consts.TEST_PRECISION)

    B = dispersion_helper._dispersion_B(k)

    assert mp.almosteq(B.imag, 0.0, abs_eps=1e-6), f"Coefficient B is not real for params: {params}"

def test_dispersion_relation(material_parameters, wave_parameters, branch):
    """Example test that combines material and wave parameters."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]


    dispersion_helper = DispersionHelper(rho = rho, lam=lam, mu=mu, nu=nu,
                            J=J, lam_c=lam_c, mu_c=mu_c, nu_c=nu_c, digits_precision=consts.TEST_PRECISION)

    c_value = dispersion_helper.c_pm(k, branch)
    cancelling_branch = mp.re(dispersion_helper._dispersion_B(k)) < 0 and branch < 0 or mp.re(dispersion_helper._dispersion_B(k)) >= 0 and branch >= 0
    B2 = dispersion_helper._dispersion_B(k)**2
    AC4 = 4 * dispersion_helper._dispersion_A(k) * dispersion_helper._dispersion_C(k)
    def precision_info():
        return f"mp.dps = {mp.dps}, rho type: {type(dispersion_helper.rho)}"

    assert mp.almosteq(dispersion_helper._dispersion(k, c_value), 0, abs_eps = 1e-6), f"Dispersion relation not satisfied for wave params: {wave_parameters}, branch: {branch}. \
        Cancelling branch: {cancelling_branch}. A: {dispersion_helper._dispersion_A(k)}, B: {dispersion_helper._dispersion_B(k)}, C: {dispersion_helper._dispersion_C(k)}, \
        c_value: {c_value}, B^2: {B2}, 4AC: {AC4}. \
        A c^2: {dispersion_helper._dispersion_A(k) * c_value**2}, B c: {dispersion_helper._dispersion_B(k) * c_value}, C: {dispersion_helper._dispersion_C(k)}. \
        A c^2 + B c: {dispersion_helper._dispersion_A(k) * c_value**2 + dispersion_helper._dispersion_B(k) * c_value}. \
            Precision info: {precision_info()}"
    
def test_cpm_prime(material_parameters, wave_parameters, branch):
    """Test that the numerical derivative of c_pm is consistent."""
    params = material_parameters | wave_parameters
    rho = params["rho"]
    lam = params["lam"]
    mu = params["mu"]
    nu = params["nu"]
    J = params["J"]
    lam_c = params["lam_c"]
    mu_c = params["mu_c"]
    nu_c = params["nu_c"]

    k = params["k"]

    dispersion_helper = DispersionHelper(rho = rho, lam=lam, mu=mu, nu=nu,
                            J=J, lam_c=lam_c, mu_c=mu_c, nu_c=nu_c, digits_precision=consts.TEST_PRECISION)
    
    h = mp.mpf('1e-8')
    c_plus_h = dispersion_helper.c_pm(k + h, branch)
    c_minus_h = dispersion_helper.c_pm(k - h, branch)
    numerical_derivative = (c_plus_h - c_minus_h) / (2 * h)

    analytical_derivative = dispersion_helper.c_pm_prime(k, branch)

    assert mp.almosteq(numerical_derivative, analytical_derivative, abs_eps=1e-6), \
        f"Numerical derivative {numerical_derivative} does not match analytical derivative {analytical_derivative} for params: {params}, branch: {branch}"