"""
Test the greens_wrapper module.

Tests the unified Green's function interface by comparing:
- Fortran backend (fast, vectorized, double precision)
- Python backend (slow, scalar, arbitrary precision)

Comparisons are done up to machine tolerance (double precision limits).
"""

from __future__ import annotations

import numpy as np
import pytest
from mpmath import mp

from cosserat_solver import consts
from cosserat_solver._integrator_core_wrapper import IntegratorFortran
from cosserat_solver.greens_wrapper import (
    FORTRAN_AVAILABLE,
    evaluate_greens_fortran,
    evaluate_greens_python,
)
from cosserat_solver.integrator import Integrator

# Test spatial positions
POSITIONS = [
    np.array([0.5, 0.3]),
    np.array([1.0, 0.0]),
    np.array([2.0, 1.5]),
    np.array([0.1, 0.2]),
]

# Test frequencies (real values for fortran, converted to complex for python)
FREQUENCIES = [
    10.0,
    50.0,
    100.0,
    500.0,
]


@pytest.fixture(params=POSITIONS)
def position(request):
    """Fixture providing test positions."""
    return request.param


@pytest.fixture(params=FREQUENCIES)
def frequency(request):
    """Fixture providing test frequencies."""
    return request.param


# Determine which backends are available
AVAILABLE_BACKENDS = ["python"]
if FORTRAN_AVAILABLE:
    AVAILABLE_BACKENDS.insert(0, "fortran")


@pytest.fixture(params=AVAILABLE_BACKENDS)
def backend_type(request):
    """Fixture parametrizing over available backends."""
    return request.param


@pytest.fixture(params=[2])
def dim(request):
    """Fixture providing problem dimension (currently only 2D)."""
    return request.param


@pytest.fixture
def evaluate_greens(backend_type, dim, material_parameters):
    """
    Fixture providing the appropriate Green's function evaluator.

    Returns a function that takes (position, frequency) and returns G(3x3).
    """
    # Convert mpmath material parameters to float
    material_params = {
        "rho": float(material_parameters["rho"]),
        "lam": float(material_parameters["lam"]),
        "mu": float(material_parameters["mu"]),
        "nu": float(material_parameters["nu"]),
        "J": float(material_parameters["J"]),
        "lam_c": float(material_parameters["lam_c"]),
        "mu_c": float(material_parameters["mu_c"]),
        "nu_c": float(material_parameters["nu_c"]),
    }

    if backend_type == "fortran":

        def _evaluate(position, frequency):
            omega_array = np.array([frequency])
            result = evaluate_greens_fortran(
                position, dim, omega_array, material_params
            )
            return result[0]

        return _evaluate

    # python
    def _evaluate(position, frequency):
        return evaluate_greens_python(
            position,
            dim,
            frequency,
            material_parameters,
            digits_precision=consts.TEST_PRECISION,
        )

    return _evaluate


class TestGreensWrapperBackends:
    """Tests for Green's function evaluation on available backends."""

    def test_greens_single_frequency(self, evaluate_greens, position, frequency):
        """Test evaluation at a single frequency."""
        G = evaluate_greens(position, frequency)

        # Check shape
        assert G.shape == (3, 3), f"Expected shape (3, 3), got {G.shape}"

        # Check dtype
        assert G.dtype == np.complex128

        # Check that result contains finite values (no NaN or Inf)
        assert np.all(np.isfinite(G)), "Green's function contains NaN or Inf"

    def test_greens_zero_frequency(self, evaluate_greens, position):
        """
        Test behavior at very low frequency (near zero).

        At omega=0, Green's function may have special behavior.
        """
        freq = 1e-6
        G = evaluate_greens(position, freq)

        assert G.shape == (3, 3)
        # May be very large but should be finite
        assert np.all(np.isfinite(G)), "Green's function contains NaN or Inf"

    def test_greens_high_frequency(self, evaluate_greens, position):
        """Test evaluation at high frequency."""
        freq = 1e4
        G = evaluate_greens(position, freq)

        assert G.shape == (3, 3)
        assert np.all(np.isfinite(G)), "Green's function contains NaN or Inf"

    def test_greens_different_positions(self, evaluate_greens, frequency):
        """Test that different positions give different Green's functions."""
        positions = POSITIONS[:3]  # Use first 3 positions
        greens = []

        for pos in positions:
            G = evaluate_greens(pos, frequency)
            greens.append(G)

        # Check that at least some values differ between positions
        differences = []
        for i in range(len(greens) - 1):
            diff = np.max(np.abs(greens[i] - greens[i + 1]))
            differences.append(diff)

        # At least one difference should be non-negligible
        max_diff = max(differences)
        assert max_diff > 1e-10, (
            "Green's functions at different positions are identical"
        )


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran backend not available")
class TestGreensWrapperFortranVectorized:
    """Tests specific to Fortran's vectorized capability."""

    def test_greens_fortran_multiple_frequencies(
        self, material_parameters, dim, position
    ):
        """Test Fortran vectorized evaluation at multiple frequencies."""
        material_params = {
            "rho": float(material_parameters["rho"]),
            "lam": float(material_parameters["lam"]),
            "mu": float(material_parameters["mu"]),
            "nu": float(material_parameters["nu"]),
            "J": float(material_parameters["J"]),
            "lam_c": float(material_parameters["lam_c"]),
            "mu_c": float(material_parameters["mu_c"]),
            "nu_c": float(material_parameters["nu_c"]),
        }

        omega_array = np.array(FREQUENCIES)
        result = evaluate_greens_fortran(position, dim, omega_array, material_params)

        # Check shape
        assert result.shape == (len(FREQUENCIES), 3, 3)

        # Check dtype
        assert result.dtype == np.complex128

        # Check that all results contain finite values
        assert np.all(np.isfinite(result)), "Green's function contains NaN or Inf"


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran backend not available")
class TestGreensWrapperCrossBackend:
    """Tests comparing Fortran and Python backends."""

    def test_fortran_python_consistency(
        self, material_parameters, dim, position, frequency
    ):
        """
        Verify that Fortran and Python backends give consistent results.

        This is a comprehensive test covering multiple material sets,
        positions, and frequencies.
        """
        material_params = {
            "rho": float(material_parameters["rho"]),
            "lam": float(material_parameters["lam"]),
            "mu": float(material_parameters["mu"]),
            "nu": float(material_parameters["nu"]),
            "J": float(material_parameters["J"]),
            "lam_c": float(material_parameters["lam_c"]),
            "mu_c": float(material_parameters["mu_c"]),
            "nu_c": float(material_parameters["nu_c"]),
        }

        def _compute_integrals_fortran_python():
            normx = float(np.linalg.norm(position))
            normx_mp = mp.mpf(normx)
            omega_complex = complex(frequency)
            normx_complex = complex(normx)

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

            integrator_python = Integrator(
                rho=material_parameters["rho"],
                lam=material_parameters["lam"],
                mu=material_parameters["mu"],
                nu=material_parameters["nu"],
                J=material_parameters["J"],
                lam_c=material_parameters["lam_c"],
                mu_c=material_parameters["mu_c"],
                nu_c=material_parameters["nu_c"],
                digits_precision=consts.TEST_PRECISION,
            )

            def _branch_integrals_fortran(omega_value, normx_value, branch):
                return {
                    "integral_3_0": integrator_fortran.integral_3_0(
                        omega_value, normx_value, branch
                    ),
                    "integral_3_2": integrator_fortran.integral_3_2(
                        omega_value, normx_value, branch
                    ),
                    "integral_2_1": integrator_fortran.integral_2_1(
                        omega_value, normx_value, branch
                    ),
                    "integral_1_0": integrator_fortran.integral_1_0(
                        omega_value, normx_value, branch
                    ),
                }

            def _branch_integrals_python(omega_value, normx_value, branch):
                return {
                    "integral_3_0": integrator_python.integral_3_0(
                        normx_value, omega_value, branch
                    ),
                    "integral_3_2": integrator_python.integral_3_2(
                        normx_value, omega_value, branch
                    ),
                    "integral_2_1": integrator_python.integral_2_1(
                        normx_value, omega_value, branch
                    ),
                    "integral_1_0": integrator_python.integral_1_0(
                        normx_value, omega_value, branch
                    ),
                }

            x_2d = [float(position[0]), float(position[1])]
            greens_fortran = {
                "P": np.array(
                    integrator_fortran.greens_x_omega_P(x_2d, omega_complex),
                    dtype=np.complex128,
                ),
                "plus": np.array(
                    integrator_fortran.greens_x_omega_plus(x_2d, omega_complex),
                    dtype=np.complex128,
                ),
                "minus": np.array(
                    integrator_fortran.greens_x_omega_minus(x_2d, omega_complex),
                    dtype=np.complex128,
                ),
            }
            greens_python = {
                "P": integrator_python.greens_x_omega_P(position, frequency),
                "plus": integrator_python.greens_x_omega_plus(position, frequency),
                "minus": integrator_python.greens_x_omega_minus(position, frequency),
            }

            integrals_fortran = {
                "plus": _branch_integrals_fortran(
                    omega_complex, normx_complex, consts.PLUS_BRANCH
                ),
                "minus": _branch_integrals_fortran(
                    omega_complex, normx_complex, -consts.PLUS_BRANCH
                ),
            }
            integrals_python = {
                "plus": _branch_integrals_python(
                    frequency, normx_mp, consts.PLUS_BRANCH
                ),
                "minus": _branch_integrals_python(
                    frequency, normx_mp, -consts.PLUS_BRANCH
                ),
            }

            return integrals_fortran, integrals_python, greens_fortran, greens_python

        # Get results from both backends
        G_fortran = evaluate_greens_fortran(
            position, dim, np.array([frequency]), material_params
        )[0]
        G_python = evaluate_greens_python(
            position,
            dim,
            frequency,
            material_parameters,
            digits_precision=consts.TEST_PRECISION,
        )

        # Compare with relaxed tolerance (numerical precision + integration error)
        eps = np.finfo(np.float64).eps
        abs_tol = eps * 1e3
        rel_tol = eps * 1e4

        for i in range(3):
            for j in range(3):
                f_val = complex(G_fortran[i, j])
                p_val = complex(G_python[i, j])

                # Skip very small values
                if abs(f_val) < abs_tol and abs(p_val) < abs_tol:
                    continue

                max_mag = max(abs(f_val), abs(p_val))
                error = abs(f_val - p_val)
                max_error = max(abs_tol, rel_tol * max_mag)

                # Use larger tolerance for cross-backend comparison
                if error > max_error * 10:
                    (
                        integrals_fortran,
                        integrals_python,
                        greens_fortran,
                        greens_python,
                    ) = _compute_integrals_fortran_python()
                    err = (
                        f"Backend mismatch at [{i},{j}]: Fortran={f_val}, Python={p_val}"
                        f"Fortran array:\n{G_fortran}"
                        f"Python array:\n{G_python}"
                        f"Fortran integrals:\n{integrals_fortran}"
                        f"Python integrals:\n{integrals_python}"
                        f"Fortran components:\n{greens_fortran}"
                        f"Python components:\n{greens_python}"
                    )
                    raise AssertionError(err)
