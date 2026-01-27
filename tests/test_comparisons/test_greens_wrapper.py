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

from cosserat_solver.greens_wrapper import (
    FORTRAN_AVAILABLE,
    evaluate_greens_fortran,
    evaluate_greens_python,
)

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


@pytest.fixture
def evaluate_greens(backend_type, material_parameters):
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
            result = evaluate_greens_fortran(position, omega_array, material_params)
            return result[0]

        return _evaluate

    # python
    def _evaluate(position, frequency):
        return evaluate_greens_python(position, frequency, material_parameters)

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

    def test_greens_fortran_multiple_frequencies(self, material_parameters, position):
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
        result = evaluate_greens_fortran(position, omega_array, material_params)

        # Check shape
        assert result.shape == (len(FREQUENCIES), 3, 3)

        # Check dtype
        assert result.dtype == np.complex128

        # Check that all results contain finite values
        assert np.all(np.isfinite(result)), "Green's function contains NaN or Inf"


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran backend not available")
class TestGreensWrapperCrossBackend:
    """Tests comparing Fortran and Python backends."""

    def test_fortran_python_consistency(self, material_parameters, position, frequency):
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

        # Get results from both backends
        G_fortran = evaluate_greens_fortran(
            position, np.array([frequency]), material_params
        )[0]
        G_python = evaluate_greens_python(position, frequency, material_parameters)

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
                assert error <= max_error * 10, (
                    f"Backend mismatch at [{i},{j}]: Fortran={f_val}, Python={p_val}"
                )
