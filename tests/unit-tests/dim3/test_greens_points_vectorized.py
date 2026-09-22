"""
Tests for the batched (points x frequencies) Green's function backend.

These validate the new `evaluate_greens_points_fortran` dispatch (and the
underlying `*_points_vectorized` Python/C/Fortran path) against the existing
single-point `evaluate_greens_fortran`. The single-point path is the trusted
reference: it uses the same quad-precision Fortran core, so the batched result
must match it to machine precision.

The layout anchor (B1) uses points NOT on the x-axis so any axis transposition
or point/frequency swap in the buffer layout fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosserat_solver import consts
from cosserat_solver.greens_wrapper import (
    FORTRAN_AVAILABLE,
    evaluate_greens_fortran,
    evaluate_greens_points_fortran,
)

pytestmark = pytest.mark.skipif(
    not FORTRAN_AVAILABLE, reason="Fortran backend not available"
)


MATERIAL_PARAMS = {
    "rho": 1.0e3,
    "lam": 1.0e5,
    "mu": 1.0e5,
    "nu": 1.0e4,
    "J": 1.0,
    "lam_c": 1.0e5,
    "mu_c": 1.0e5,
    "nu_c": 1.0e4,
}

# Asymmetric points, deliberately off the x-axis, so a transposition or a
# point/frequency swap in the layout would break the comparison.
POINTS = np.array(
    [
        [1.3, -0.7, 2.1],
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0],
        [-0.4, 1.2, -0.9],
    ]
)

OMEGA_ARRAY = np.array([10.0, 55.0, 123.0, 400.0, 777.0])

MATERIAL_TYPES = [
    pytest.param(consts.MATERIAL_TYPE_COSSERAT, id="cosserat"),
    pytest.param(consts.MATERIAL_TYPE_ELASTIC, id="elastic"),
]


@pytest.mark.parametrize("material_type", MATERIAL_TYPES)
def test_batched_matches_single_point(material_type):
    """B1 (layout anchor): batched result equals stacked single-point calls."""
    batched = evaluate_greens_points_fortran(
        POINTS,
        consts.DIMENSION_3D,
        OMEGA_ARRAY,
        MATERIAL_PARAMS,
        material_type=material_type,
    )
    assert batched.shape == (len(POINTS), len(OMEGA_ARRAY), 6, 6)

    for p, point in enumerate(POINTS):
        expected = evaluate_greens_fortran(
            point,
            consts.DIMENSION_3D,
            OMEGA_ARRAY,
            MATERIAL_PARAMS,
            material_type=material_type,
        )  # (n_omega, 6, 6)
        np.testing.assert_allclose(batched[p], expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("material_type", MATERIAL_TYPES)
def test_openmp_modes_agree(material_type):
    """B2: forced-on, forced-off, and auto OpenMP give identical results."""
    auto = evaluate_greens_points_fortran(
        POINTS, consts.DIMENSION_3D, OMEGA_ARRAY, MATERIAL_PARAMS, material_type
    )
    forced_on = evaluate_greens_points_fortran(
        POINTS,
        consts.DIMENSION_3D,
        OMEGA_ARRAY,
        MATERIAL_PARAMS,
        material_type,
        force_use_openmp=True,
    )
    forced_off = evaluate_greens_points_fortran(
        POINTS,
        consts.DIMENSION_3D,
        OMEGA_ARRAY,
        MATERIAL_PARAMS,
        material_type,
        force_no_openmp=True,
    )
    np.testing.assert_array_equal(auto, forced_on)
    np.testing.assert_array_equal(auto, forced_off)


@pytest.mark.parametrize("material_type", MATERIAL_TYPES)
def test_both_openmp_flags_raises(material_type):
    """B2: both force flags set raises ValueError at the Python layer."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate_greens_points_fortran(
            POINTS,
            consts.DIMENSION_3D,
            OMEGA_ARRAY,
            MATERIAL_PARAMS,
            material_type,
            force_use_openmp=True,
            force_no_openmp=True,
        )


@pytest.mark.parametrize("material_type", MATERIAL_TYPES)
def test_single_point_single_frequency(material_type):
    """B3: n_points == 1 and n_omega == 1 edge cases."""
    point = np.array([[1.3, -0.7, 2.1]])
    omega = np.array([123.0])
    batched = evaluate_greens_points_fortran(
        point, consts.DIMENSION_3D, omega, MATERIAL_PARAMS, material_type
    )
    assert batched.shape == (1, 1, 6, 6)
    expected = evaluate_greens_fortran(
        point[0], consts.DIMENSION_3D, omega, MATERIAL_PARAMS, material_type
    )
    np.testing.assert_allclose(batched[0], expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("material_type", MATERIAL_TYPES)
def test_invalid_points_shape_raises(material_type):
    """B4: wrong points shape raises ValueError."""
    bad_points = np.array([1.0, 2.0, 3.0])  # 1-D, not (n_points, 3)
    with pytest.raises(ValueError, match="points must have shape"):
        evaluate_greens_points_fortran(
            bad_points, consts.DIMENSION_3D, OMEGA_ARRAY, MATERIAL_PARAMS, material_type
        )


@pytest.mark.parametrize("material_type", MATERIAL_TYPES)
def test_empty_omega_raises(material_type):
    """B4: empty omega raises ValueError."""
    with pytest.raises(ValueError, match="at least one frequency"):
        evaluate_greens_points_fortran(
            POINTS,
            consts.DIMENSION_3D,
            np.array([]),
            MATERIAL_PARAMS,
            material_type,
        )
