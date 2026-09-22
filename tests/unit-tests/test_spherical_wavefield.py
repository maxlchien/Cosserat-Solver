"""
Tests for the spherical wavefield radial computation.

Covers:
- get_snapshot_indices and _resolve_t0: pure-Python, no backend.
- compute_source_radial_tensor: validated against generate_trace, plus
  chunking invariance, snapshot selection, and input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosserat_solver import consts
from cosserat_solver.greens_wrapper import (
    FORTRAN_AVAILABLE,
    evaluate_greens_fortran,
)
from cosserat_solver.ricker import Ricker3D
from cosserat_solver.spherical_wavefield import (
    _resolve_t0,
    block_rotations,
    compute_source_radial_tensor,
    generate_spherical_grid,
    get_snapshot_indices,
    rotate_and_contract,
    rotation_matrices,
    write_spherical_wavefield,
)
from cosserat_solver.trace_generator import channels_3d, generate_trace, get_prefix

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

# Small grid; quad precision is slow.
SIM_PARAMS = {
    "N": 64,
    "dt": 0.002,
    "extension_factor": 4,
    "refinement_factor": 4,
}


def make_source():
    """Ricker3D at the origin with an asymmetric direction vector."""
    return Ricker3D(
        {
            "f0": 10.0,
            "f": [1.0, -0.5, 0.3],
            "fc": [0.2, 0.7, -0.4],
            "location": [0.0, 0.0, 0.0],
        }
    )


# ---------------------------------------------------------------------------
# get_snapshot_indices
# ---------------------------------------------------------------------------


def test_get_snapshot_indices_default_all():
    """Interval 1 (default) selects every sample."""
    idx = get_snapshot_indices({"N": 64})
    np.testing.assert_array_equal(idx, np.arange(64))


def test_get_snapshot_indices_stride():
    """Interval k selects every k-th of the N-grid."""
    idx = get_snapshot_indices({"N": 64, "steps_per_snapshot": 10})
    np.testing.assert_array_equal(idx, np.arange(0, 64, 10))


def test_get_snapshot_indices_missing_N():
    with pytest.raises(ValueError, match="must contain 'N'"):
        get_snapshot_indices({"steps_per_snapshot": 2})


@pytest.mark.parametrize(
    ("interval", "match"),
    [
        (0, ">= 1"),
        (-3, ">= 1"),
        (1.5, "must be an integer"),
        (True, "not a bool"),
        (False, "not a bool"),
    ],
)
def test_get_snapshot_indices_bad_interval(interval, match):
    """A bad steps_per_snapshot (0, negative, non-integer, bool) raises."""
    with pytest.raises(ValueError, match=match):
        get_snapshot_indices({"N": 64, "steps_per_snapshot": interval})


# ---------------------------------------------------------------------------
# _resolve_t0
# ---------------------------------------------------------------------------


def test_resolve_t0_explicit():
    """An explicit t0 is returned verbatim."""
    assert _resolve_t0({"t0": 1.25}, []) == 1.25


def test_resolve_t0_from_sources_min():
    """Without explicit t0, the minimum of the sources' t0() is used."""
    s1 = Ricker3D({"f0": 10.0})  # t0 = -1.2/10
    s2 = Ricker3D({"f0": 5.0})  # t0 = -1.2/5 (more negative)
    assert _resolve_t0({}, [s1, s2]) == min(s1.t0(), s2.t0())


def test_resolve_t0_no_sources_no_t0_raises():
    with pytest.raises(ValueError, match="no explicit 't0'"):
        _resolve_t0({}, [])


# ---------------------------------------------------------------------------
# compute_source_radial_tensor validation (no backend needed)
# ---------------------------------------------------------------------------


def test_radial_tensor_nonpositive_radii_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        compute_source_radial_tensor(
            np.array([1.0, 0.0, 2.0]),
            {**SIM_PARAMS, "t0": 0.0},
            MATERIAL_PARAMS,
            make_source(),
            consts.MATERIAL_TYPE_COSSERAT,
        )


def test_radial_tensor_off_origin_source_raises():
    source = Ricker3D({"f0": 10.0, "location": [1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="source at the origin"):
        compute_source_radial_tensor(
            np.array([1.0, 2.0]),
            {**SIM_PARAMS, "t0": 0.0},
            MATERIAL_PARAMS,
            source,
            consts.MATERIAL_TYPE_COSSERAT,
        )


def test_radial_tensor_dim2_raises():
    with pytest.raises(ValueError, match="3D only"):
        compute_source_radial_tensor(
            np.array([1.0, 2.0]),
            {**SIM_PARAMS, "t0": 0.0},
            MATERIAL_PARAMS,
            make_source(),
            consts.MATERIAL_TYPE_COSSERAT,
            dim=consts.DIMENSION_2D,
        )


def test_radial_tensor_no_fortran_raises():
    with pytest.raises(ValueError, match="require the Fortran backend"):
        compute_source_radial_tensor(
            np.array([1.0, 2.0]),
            {**SIM_PARAMS, "t0": 0.0},
            MATERIAL_PARAMS,
            make_source(),
            consts.MATERIAL_TYPE_COSSERAT,
            use_fortran=False,
        )


# ---------------------------------------------------------------------------
# compute_source_radial_tensor numerics (Fortran backend)
# ---------------------------------------------------------------------------

pytest_fortran = pytest.mark.skipif(
    not FORTRAN_AVAILABLE, reason="Fortran backend not available"
)


@pytest_fortran
@pytest.mark.parametrize(
    "material_type",
    [
        pytest.param(consts.MATERIAL_TYPE_COSSERAT, id="cosserat"),
        pytest.param(consts.MATERIAL_TYPE_ELASTIC, id="elastic"),
    ],
)
def test_radial_tensor_matches_trace(material_type):
    """The radial tensor on +x, contracted with source.direction(), matches
    generate_trace channel-by-channel at receiver (r, 0, 0)."""
    source = make_source()
    r = 50.0
    sim = {**SIM_PARAMS, "t0": source.t0()}

    times, tensor = compute_source_radial_tensor(
        np.array([r]),
        sim,
        MATERIAL_PARAMS,
        source,
        material_type,
    )
    # Contract the single-radius tensor with the source direction vector.
    v = np.einsum("tij,j->ti", tensor[0], source.direction())  # (N, 6)

    trace_times, traces = generate_trace(
        {"location": np.array([r, 0.0, 0.0])},
        consts.DIMENSION_3D,
        material_type,
        MATERIAL_PARAMS,
        [source],
        sim,
    )
    prefix = get_prefix(sim["dt"])
    expected = np.stack(
        [traces[f"{prefix}{channel}"] for channel in channels_3d], axis=1
    )  # (N, 6)

    np.testing.assert_allclose(times, trace_times, rtol=1e-12)
    peak = np.max(np.abs(expected))
    atol = 1e-10 * peak if peak > 0 else 1e-12
    np.testing.assert_allclose(v, expected, rtol=1e-9, atol=atol)


@pytest_fortran
def test_radial_tensor_chunking_invariance():
    """radial_chunk_size in {None, 1, 2} give bit-identical tensors."""
    source = make_source()
    radii = np.array([10.0, 30.0, 70.0])
    sim = {**SIM_PARAMS, "t0": source.t0()}

    _, full = compute_source_radial_tensor(
        radii, sim, MATERIAL_PARAMS, source, consts.MATERIAL_TYPE_COSSERAT
    )
    for chunk in (1, 2):
        _, chunked = compute_source_radial_tensor(
            radii,
            sim,
            MATERIAL_PARAMS,
            source,
            consts.MATERIAL_TYPE_COSSERAT,
            radial_chunk_size=chunk,
        )
        np.testing.assert_array_equal(full, chunked)


@pytest_fortran
def test_radial_tensor_snapshot_selection():
    """time_indices selects exactly those rows of the full tensor."""
    source = make_source()
    radii = np.array([20.0, 40.0])
    sim = {**SIM_PARAMS, "t0": source.t0()}

    full_times, full = compute_source_radial_tensor(
        radii, sim, MATERIAL_PARAMS, source, consts.MATERIAL_TYPE_COSSERAT
    )

    idx = get_snapshot_indices({**sim, "steps_per_snapshot": 8})
    sel_times, selected = compute_source_radial_tensor(
        radii,
        sim,
        MATERIAL_PARAMS,
        source,
        consts.MATERIAL_TYPE_COSSERAT,
        time_indices=idx,
    )

    assert selected.shape == (2, len(idx), 6, 6)
    np.testing.assert_array_equal(sel_times, full_times[idx])
    np.testing.assert_array_equal(selected, full[:, idx])


# ---------------------------------------------------------------------------
# rotation_matrices / block_rotations / generate_spherical_grid (pure numpy)
# ---------------------------------------------------------------------------

THETAS = [0.0, np.pi / 2, np.pi]
PHIS = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]


def _r_hat(theta, phi):
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def test_rotation_matrices_properties():
    """Q maps +x to r_hat, is orthogonal, and proper (det = +1) everywhere,
    including the poles and -x."""
    theta = np.array(THETAS)
    phi = np.array(PHIS)
    Q = rotation_matrices(theta, phi)
    assert Q.shape == (len(THETAS), len(PHIS), 3, 3)

    xhat = np.array([1.0, 0.0, 0.0])
    for i, th in enumerate(THETAS):
        for j, ph in enumerate(PHIS):
            q = Q[i, j]
            np.testing.assert_allclose(q @ xhat, _r_hat(th, ph), atol=1e-14)
            np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-14)
            np.testing.assert_allclose(np.linalg.det(q), 1.0, atol=1e-14)


def test_rotation_matrices_non_1d_raises():
    with pytest.raises(ValueError, match="theta must be a 1D array"):
        rotation_matrices(np.zeros((2, 2)), np.array([0.0]))
    with pytest.raises(ValueError, match="phi must be a 1D array"):
        rotation_matrices(np.array([0.0]), np.zeros((2, 2)))


def test_block_rotations_is_blockdiag():
    """block_rotations builds blockdiag(Q, Q) with zero off-diagonal blocks."""
    Q = rotation_matrices(np.array(THETAS), np.array(PHIS))
    D = block_rotations(Q)
    assert D.shape == (*Q.shape[:-2], 6, 6)
    np.testing.assert_array_equal(D[..., :3, :3], Q)
    np.testing.assert_array_equal(D[..., 3:, 3:], Q)
    np.testing.assert_array_equal(D[..., :3, 3:], 0.0)
    np.testing.assert_array_equal(D[..., 3:, :3], 0.0)


def test_block_rotations_bad_shape_raises():
    with pytest.raises(ValueError, match=r"trailing shape \(3, 3\)"):
        block_rotations(np.zeros((4, 4)))


def test_generate_spherical_grid_formulas():
    """coordinates follow x/y/z spherical formulas with (r, theta, phi) axis
    order, and directions are the corresponding unit vectors."""
    radii = np.array([2.0, 5.0])
    theta = np.array([0.3, 1.1, np.pi - 0.2])
    phi = np.array([0.0, 1.7, 4.9])

    directions, coordinates = generate_spherical_grid(radii, theta, phi)
    assert directions.shape == (len(theta), len(phi), 3)
    assert coordinates.shape == (len(radii), len(theta), len(phi), 3)

    for i, th in enumerate(theta):
        for j, ph in enumerate(phi):
            np.testing.assert_allclose(directions[i, j], _r_hat(th, ph), atol=1e-14)
            for k, r in enumerate(radii):
                np.testing.assert_allclose(
                    coordinates[k, i, j], r * _r_hat(th, ph), atol=1e-13
                )


@pytest.mark.parametrize(
    ("radii", "theta", "phi", "match"),
    [
        (np.array([-1.0]), np.array([0.5]), np.array([0.5]), "strictly positive"),
        (np.array([1.0]), np.array([-0.1]), np.array([0.5]), r"\[0, pi\]"),
        (np.array([1.0]), np.array([np.pi + 0.1]), np.array([0.5]), r"\[0, pi\]"),
        (np.array([1.0]), np.array([0.5]), np.array([-0.1]), r"\[0, 2\*pi\)"),
        (np.array([1.0]), np.array([0.5]), np.array([2 * np.pi]), r"\[0, 2\*pi\)"),
    ],
)
def test_generate_spherical_grid_validation(radii, theta, phi, match):
    with pytest.raises(ValueError, match=match):
        generate_spherical_grid(radii, theta, phi)


# ---------------------------------------------------------------------------
# Rotation covariance and end-to-end reconstruction (Fortran backend)
# ---------------------------------------------------------------------------


@pytest_fortran
@pytest.mark.parametrize(
    "material_type",
    [
        pytest.param(consts.MATERIAL_TYPE_COSSERAT, id="cosserat"),
        pytest.param(consts.MATERIAL_TYPE_ELASTIC, id="elastic"),
    ],
)
def test_rotation_covariance(material_type):
    """G(r * r_hat, omega) == D @ G([r,0,0], omega) @ D.T for proper rotations D,
    validating the reconstruction premise against the real backend."""
    omega = np.array([12.0, 87.0, 260.0])
    r = 40.0
    theta = np.array([0.7])
    phi = np.array([2.3])
    Q = rotation_matrices(theta, phi)[0, 0]
    D = block_rotations(Q)

    x_axis = np.array([r, 0.0, 0.0])
    x_rot = r * _r_hat(theta[0], phi[0])

    G_axis = evaluate_greens_fortran(
        x_axis, consts.DIMENSION_3D, omega, MATERIAL_PARAMS, material_type
    )
    G_rot = evaluate_greens_fortran(
        x_rot, consts.DIMENSION_3D, omega, MATERIAL_PARAMS, material_type
    )

    expected = np.einsum("ab,wbc,dc->wad", D, G_axis, D)  # D @ G @ D.T
    np.testing.assert_allclose(G_rot, expected, rtol=1e-9, atol=1e-12)


@pytest_fortran
def test_end_to_end_off_axis_matches_trace():
    """Full rotate-and-contract time series at an off-axis (r, theta, phi)
    matches generate_trace at the Cartesian receiver r * r_hat."""
    source = make_source()
    r = 45.0
    theta_val, phi_val = 0.9, 2.1
    sim = {**SIM_PARAMS, "t0": source.t0()}

    # Radial tensor on +x, then reconstruct at the single off-axis direction.
    times, tensor = compute_source_radial_tensor(
        np.array([r]), sim, MATERIAL_PARAMS, source, consts.MATERIAL_TYPE_COSSERAT
    )
    Q = rotation_matrices(np.array([theta_val]), np.array([phi_val]))[0, 0]
    D = block_rotations(Q)[None, :, :]  # (n_dir=1, 6, 6)
    f = source.direction()

    n_t = tensor.shape[1]
    v = np.empty((n_t, 6))
    for k in range(n_t):
        v[k] = rotate_and_contract(tensor[:, k], D, f)[0, 0]  # (radius0, dir0, 6)

    receiver = r * _r_hat(theta_val, phi_val)
    trace_times, traces = generate_trace(
        {"location": receiver},
        consts.DIMENSION_3D,
        consts.MATERIAL_TYPE_COSSERAT,
        MATERIAL_PARAMS,
        [source],
        sim,
    )
    prefix = get_prefix(sim["dt"])
    expected = np.stack(
        [traces[f"{prefix}{channel}"] for channel in channels_3d], axis=1
    )

    np.testing.assert_allclose(times, trace_times, rtol=1e-12)
    peak = np.max(np.abs(expected))
    atol = 1e-10 * peak if peak > 0 else 1e-12
    np.testing.assert_allclose(v, expected, rtol=1e-9, atol=atol)


# ---------------------------------------------------------------------------
# Full spherical grid vs Cartesian (HDF5 writer end-to-end, Fortran backend)
# ---------------------------------------------------------------------------


@pytest_fortran
def test_full_spherical_grid_matches_cartesian(tmp_path):
    """The written HDF5 wavefield over a full spherical grid equals, point by
    point and sample by sample, a direct Cartesian generate_trace at each grid
    node's coordinates.

    The grid spans both poles (theta = 0, pi) and all four quadrants in phi, so
    the pole degeneracy (all phi collapse to the same +/-z point) and the -x
    direction are exercised. Every (r, theta, phi) node is reconstructed by the
    rotation law; the reference is an independent Cartesian solve.
    """
    h5py = pytest.importorskip("h5py")

    source = make_source()
    radii = np.array([30.0, 60.0])
    theta = np.array([0.0, np.pi / 2, np.pi])
    phi = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    sim = {**SIM_PARAMS, "t0": source.t0()}

    out_path = str(tmp_path / "wavefield.h5")
    meta = write_spherical_wavefield(
        out_path,
        radii,
        theta,
        phi,
        consts.DIMENSION_3D,
        consts.MATERIAL_TYPE_COSSERAT,
        MATERIAL_PARAMS,
        [source],
        sim,
        # Every sample is a snapshot so we can compare the whole time series.
        write_xdmf=False,
        compression=None,
        close_phi_seam=False,
    )

    assert meta["n_snapshots"] == SIM_PARAMS["N"]
    assert meta["field_shape"] == (len(radii), len(theta), len(phi), 3)

    with h5py.File(out_path, "r") as h5:
        coordinates = h5["coordinates"][:]  # (n_r, n_theta, n_phi, 3)
        h5_times = h5["times"][:]
        step_names = meta["step_names"]
        # Reassemble the (n_snap, n_r, n_theta, n_phi, 6) block from the per-step
        # Displacement (0:3) and Rotation (3:6) datasets.
        disp = np.stack([h5[s]["Displacement"][:] for s in step_names], axis=0)
        rot = np.stack([h5[s]["Rotation"][:] for s in step_names], axis=0)
    values = np.concatenate([disp, rot], axis=-1)

    prefix = get_prefix(sim["dt"])
    # Compare the reconstruction against a direct Cartesian solve at every node.
    for i in range(len(radii)):
        for j in range(len(theta)):
            for k in range(len(phi)):
                receiver = coordinates[i, j, k]
                # coordinates match r * r_hat(theta, phi).
                np.testing.assert_allclose(
                    receiver, radii[i] * _r_hat(theta[j], phi[k]), atol=1e-12
                )

                trace_times, traces = generate_trace(
                    {"location": receiver},
                    consts.DIMENSION_3D,
                    consts.MATERIAL_TYPE_COSSERAT,
                    MATERIAL_PARAMS,
                    [source],
                    sim,
                )
                expected = np.stack(
                    [traces[f"{prefix}{channel}"] for channel in channels_3d],
                    axis=1,
                )  # (N, 6)

                np.testing.assert_allclose(h5_times, trace_times, rtol=1e-12)
                peak = np.max(np.abs(expected))
                atol = 1e-10 * peak if peak > 0 else 1e-12
                np.testing.assert_allclose(
                    values[:, i, j, k, :], expected, rtol=1e-9, atol=atol
                )


@pytest_fortran
def test_close_phi_seam_wraps_first_slice(tmp_path):
    """close_phi_seam=True appends exactly one phi column (phi=2*pi) whose
    coordinates and fields duplicate the phi=0 slice, while leaving the first
    n_phi columns bit-identical to the unwrapped (close_phi_seam=False) output.
    """
    h5py = pytest.importorskip("h5py")

    source = make_source()
    radii = np.array([20.0, 50.0])
    theta = np.array([0.4, 1.5, 2.6])
    phi = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])  # full circle
    sim = {**SIM_PARAMS, "t0": source.t0(), "steps_per_snapshot": 16}
    n_phi = len(phi)

    common = {"write_xdmf": False, "compression": None}  # avoid I/O for this test

    open_path = str(tmp_path / "open.h5")
    write_spherical_wavefield(
        open_path,
        radii,
        theta,
        phi,
        consts.DIMENSION_3D,
        consts.MATERIAL_TYPE_COSSERAT,
        MATERIAL_PARAMS,
        [source],
        sim,
        close_phi_seam=False,
        **common,
    )
    closed_path = str(tmp_path / "closed.h5")
    meta = write_spherical_wavefield(
        closed_path,
        radii,
        theta,
        phi,
        consts.DIMENSION_3D,
        consts.MATERIAL_TYPE_COSSERAT,
        MATERIAL_PARAMS,
        [source],
        sim,
        close_phi_seam=True,
        **common,
    )

    assert meta["field_shape"] == (len(radii), len(theta), n_phi + 1, 3)

    with h5py.File(open_path, "r") as fo, h5py.File(closed_path, "r") as fc:
        phi_open = fo["phi"][:]
        phi_closed = fc["phi"][:]
        coords_open = fo["coordinates"][:]
        coords_closed = fc["coordinates"][:]
        step_names = meta["step_names"]

        # phi axis: original values preserved, one extra column at 2*pi.
        assert phi_closed.shape == (n_phi + 1,)
        np.testing.assert_array_equal(phi_closed[:n_phi], phi_open)
        np.testing.assert_allclose(phi_closed[-1], 2.0 * np.pi)

        # Coordinates: first n_phi columns unchanged; wrap column == phi=0 slice.
        assert coords_closed.shape == (len(radii), len(theta), n_phi + 1, 3)
        np.testing.assert_array_equal(coords_closed[:, :, :n_phi, :], coords_open)
        np.testing.assert_array_equal(
            coords_closed[:, :, -1, :], coords_closed[:, :, 0, :]
        )

        # Every snapshot field: same invariants for Displacement and Rotation.
        for step_name in step_names:
            for field in ("Displacement", "Rotation"):
                d_open = fo[step_name][field][:]
                d_closed = fc[step_name][field][:]
                assert d_closed.shape == (len(radii), len(theta), n_phi + 1, 3)
                np.testing.assert_array_equal(d_closed[:, :, :n_phi, :], d_open)
                np.testing.assert_array_equal(
                    d_closed[:, :, -1, :], d_closed[:, :, 0, :]
                )
