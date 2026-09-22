from __future__ import annotations

import os

import numpy as np
from loguru import logger

import cosserat_solver.fourier as fourier
from cosserat_solver import consts
from cosserat_solver.greens_wrapper import (
    FORTRAN_AVAILABLE,
    evaluate_greens_points_fortran,
)
from cosserat_solver.source import SourceSpectrum


def get_snapshot_indices(simulation_params: dict) -> np.ndarray:
    """
    Indices into the N-sample output time grid selected for snapshots.

    Reads ``simulation_params["steps_per_snapshot"]`` (default 1) and returns
    ``np.arange(0, N, interval)``. The interval strides the already-downsampled
    N-grid (the output trace grid), never the dense FFT grid used internally by
    :func:`cosserat_solver.fourier.cont_ifft`.

    Parameters
    ----------
    simulation_params : dict
        Must contain 'N'. Optionally contains 'steps_per_snapshot' (a positive
        integer; defaults to 1 = every sample).

    Returns
    -------
    np.ndarray
        1-D integer array of snapshot indices into the N-sample grid.

    Raises
    ------
    ValueError
        If 'N' is missing, or 'steps_per_snapshot' is a bool, non-integer, or < 1.
    """
    if "N" not in simulation_params:
        err = "simulation_params must contain 'N'."
        logger.error(err)
        raise ValueError(err)
    N = int(simulation_params["N"])

    interval = simulation_params.get("steps_per_snapshot", 1)

    # Reject bools explicitly (bool is a subclass of int).
    if isinstance(interval, bool):
        err = "steps_per_snapshot must be an integer, not a bool."
        logger.error(err)
        raise ValueError(err)

    # Require an integer value (reject e.g. 1.5).
    try:
        is_integer_valued = float(interval).is_integer()
    except (TypeError, ValueError):
        is_integer_valued = False
    if not is_integer_valued:
        err = f"steps_per_snapshot must be an integer, got {interval!r}."
        logger.error(err)
        raise ValueError(err)

    interval = int(interval)
    if interval < 1:
        err = f"steps_per_snapshot must be >= 1, got {interval}."
        logger.error(err)
        raise ValueError(err)

    return np.arange(0, N, interval)


def _resolve_t0(simulation_params: dict, sources: list[SourceSpectrum]) -> float:
    """
    Resolve the single shared ``t0`` for the wavefield time grid.

    Unlike :func:`cosserat_solver.trace_generator.generate_trace`, which copies the
    simulation parameters per source, the wavefield needs ONE time grid shared across
    all sources. If ``t0`` is provided explicitly it is used; otherwise the minimum of
    the sources' ``t0()`` is used so that, together with the extension window, every
    source's support stays inside the FFT window.

    Parameters
    ----------
    simulation_params : dict
        May contain an explicit 't0'.
    sources : list[SourceSpectrum]
        The sources sharing the grid; must be non-empty when 't0' is absent.

    Returns
    -------
    float
        The resolved start time.
    """
    if "t0" in simulation_params:
        return simulation_params["t0"]

    if not sources:
        err = "Cannot resolve t0: no explicit 't0' and no sources provided."
        logger.error(err)
        raise ValueError(err)

    t0s = [source.t0() for source in sources]
    if not np.allclose(t0s, t0s[0]):
        logger.warning(
            "Sources have differing t0() values {t0s}; using the minimum ({t0}). "
            "Consider setting an explicit 't0' in simulation_params for the shared "
            "wavefield time grid.",
            t0s=t0s,
            t0=min(t0s),
        )
    return min(t0s)


def compute_source_radial_tensor(
    radii: np.ndarray,
    simulation_params: dict,
    material_params: dict,
    source: SourceSpectrum,
    material_type: int,
    dim: int = consts.DIMENSION_3D,
    use_fortran: bool = True,
    force_use_openmp: bool = False,
    force_no_openmp: bool = False,
    time_indices: np.ndarray | None = None,
    radial_chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute the time-domain radial Green tensor for a single source.

    Along the +x radial ray, evaluates the 6x6 frequency-domain Green tensor at each
    radius (batched over radii and frequencies in one backend call per chunk),
    multiplies by the source spectrum, and inverse-transforms to time. By isotropy the
    full spherical field is later reconstructed from this radial tensor by rotation.

    Returns ``(times, tensor)`` where::

        tensor[i, k] = Re[ IFT{ G((r_i, 0, 0), omega) * spectrum(omega) }(t_k) ]

    Parameters
    ----------
    radii : np.ndarray
        Strictly-positive radii, shape (n_radii,).
    simulation_params : dict
        Fourier parameters (N, dt, extension_factor, ...). Must already contain the
        resolved 't0' for the shared grid; this function does NOT copy or modify it.
        Public callers using this function directly must set 't0' themselves.
    material_params : dict
        Material parameters (rho, lam, mu, nu, J, lam_c, mu_c, nu_c).
    source : SourceSpectrum
        Source at the origin. ``source.location()`` must be length 3 and ~zero.
    material_type : int
        Material type (Cosserat or elastic).
    dim : int, default=consts.DIMENSION_3D
        Only DIMENSION_3D is supported.
    use_fortran : bool, default=True
        Must be True; spherical wavefields require the Fortran backend.
    force_use_openmp : bool, default=False
        If True, force OpenMP parallelization even for small batches.
    force_no_openmp : bool, default=False
        If True, disable OpenMP parallelization even for large batches.
    time_indices : np.ndarray | None, default=None
        If given, only these indices of the N-sample output grid are retained (the
        snapshot stride). The full inverse FFT plus extension/refinement removal is
        performed first for a chunk, then this stride is applied, all before the next
        chunk begins.
    radial_chunk_size : int | None, default=None
        If None, all radii are processed in one chunk. Otherwise radii are processed
        in chunks of this size, bounding transient memory. The dense per-chunk
        workspace peaks at roughly ``chunk * N_dense * 36 * 100`` bytes, where
        ``N_dense`` is the oversampled FFT length
        (~N * extension_factor * refinement_factor). Chunk to keep this within budget.

    Returns
    -------
    times : np.ndarray
        Shape (n_t,). The full N-grid times if ``time_indices`` is None, else
        ``times[time_indices]``.
    tensor : np.ndarray
        Shape (n_radii, n_t, 6, 6), float64 (the real part of the transform).

    Raises
    ------
    ValueError
        If inputs are invalid (see validation below).
    """
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1:
        err = "radii must be a 1D array."
        logger.error(err)
        raise ValueError(err)
    if radii.size == 0:
        err = "radii must contain at least one radius."
        logger.error(err)
        raise ValueError(err)
    if np.any(radii <= 0.0):
        err = "radii must be strictly positive (the source sits at the origin)."
        logger.error(err)
        raise ValueError(err)

    if dim != consts.DIMENSION_3D:
        err = f"Spherical wavefields are 3D only; got dim={dim}."
        logger.error(err)
        raise ValueError(err)

    if not isinstance(source, SourceSpectrum):
        err = f"source must be a SourceSpectrum, got {type(source)}."
        logger.error(err)
        raise TypeError(err)

    location = np.asarray(source.location(), dtype=float)
    if location.shape != (3,) or not np.allclose(location, 0.0):
        err = (
            "Spherical wavefields require a source at the origin; "
            f"got location {location}."
        )
        logger.error(err)
        raise ValueError(err)

    if not (use_fortran and FORTRAN_AVAILABLE):
        err = (
            "Spherical wavefields require the Fortran backend "
            "(the Python/mpmath path is orders of magnitude too slow)."
        )
        logger.error(err)
        raise ValueError(err)

    if radial_chunk_size is not None:
        if (
            isinstance(radial_chunk_size, bool)
            or not float(radial_chunk_size).is_integer()
        ):
            err = f"radial_chunk_size must be an integer or None, got {radial_chunk_size!r}."
            logger.error(err)
            raise ValueError(err)
        radial_chunk_size = int(radial_chunk_size)
        if radial_chunk_size < 1:
            err = f"radial_chunk_size must be >= 1, got {radial_chunk_size}."
            logger.error(err)
            raise ValueError(err)

    n_radii = radii.shape[0]
    chunk = radial_chunk_size if radial_chunk_size is not None else n_radii

    full_times = fourier._time_array(simulation_params)
    times = full_times if time_indices is None else full_times[time_indices]
    n_t = times.shape[0]

    tensor = np.empty((n_radii, n_t, 6, 6), dtype=np.float64)

    for start in range(0, n_radii, chunk):
        chunk_radii = radii[start : start + chunk]
        points = np.column_stack(
            [chunk_radii, np.zeros_like(chunk_radii), np.zeros_like(chunk_radii)]
        )

        def f_hat(omega_array: np.ndarray, points: np.ndarray = points) -> np.ndarray:
            # (n_chunk, n_omega, 6, 6)
            G = evaluate_greens_points_fortran(
                points,
                dim,
                omega_array,
                material_params,
                material_type,
                force_use_openmp=force_use_openmp,
                force_no_openmp=force_no_openmp,
            )
            # Apply the source spectrum exactly once (the batched backend does not).
            return G * source.spectrum_vectorized(omega_array)[None, :, None, None]

        # axis=1 is the frequency/time axis: g_time is (n_chunk, N, 6, 6). The full
        # inverse FFT plus extension/refinement removal happens inside cont_ifft; the
        # snapshot stride is applied here, all before the next chunk starts.
        _, g_time = fourier.cont_ifft(f_hat, simulation_params, axis=1)
        tensor[start : start + chunk] = np.real(
            g_time if time_indices is None else g_time[:, time_indices]
        )

    return times, tensor


def rotation_matrices(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    r"""
    Proper rotations ``Q`` mapping the +x axis onto each ``(theta, phi)`` direction.

    Uses the closed form ``Q(theta, phi) = Rz(phi) @ Ry(theta - pi/2)``, which
    satisfies ``Q @ [1, 0, 0] == r_hat(theta, phi)`` and is always a proper rotation
    (det = +1). Built columnwise, so there are no runtime matrix products and no
    special-casing at the poles or at -x.

    With st = sin(theta), ct = cos(theta), cp = cos(phi), sp = sin(phi)::

        col0 = ( st*cp,  st*sp,  ct )   # = r_hat, the invariant Q x_hat = r_hat
        col1 = ( -sp,     cp,     0 )   # = phi_hat
        col2 = ( -ct*cp, -ct*sp,  st )  # completes the right-handed frame

    Parameters
    ----------
    theta : np.ndarray
        1-D polar angles from +z, shape (n_theta,).
    phi : np.ndarray
        1-D azimuths from +x toward +y, shape (n_phi,).

    Returns
    -------
    np.ndarray
        Shape (n_theta, n_phi, 3, 3), float64. ``Q[i, j]`` rotates +x onto
        ``r_hat(theta[i], phi[j])``.

    Raises
    ------
    ValueError
        If ``theta`` or ``phi`` is not 1-D.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if theta.ndim != 1:
        err = "theta must be a 1D array."
        logger.error(err)
        raise ValueError(err)
    if phi.ndim != 1:
        err = "phi must be a 1D array."
        logger.error(err)
        raise ValueError(err)

    # Broadcast theta over rows, phi over columns: (n_theta, n_phi).
    st = np.sin(theta)[:, None]
    ct = np.cos(theta)[:, None]
    cp = np.cos(phi)[None, :]
    sp = np.sin(phi)[None, :]
    st, ct, cp, sp = np.broadcast_arrays(st, ct, cp, sp)

    n_theta, n_phi = st.shape
    Q = np.empty((n_theta, n_phi, 3, 3), dtype=np.float64)
    zero = np.zeros_like(st)

    # Column 0 = r_hat.
    Q[..., 0, 0] = st * cp
    Q[..., 1, 0] = st * sp
    Q[..., 2, 0] = ct
    # Column 1 = phi_hat.
    Q[..., 0, 1] = -sp
    Q[..., 1, 1] = cp
    Q[..., 2, 1] = zero
    # Column 2 = theta_hat (completes right-handed frame).
    Q[..., 0, 2] = -ct * cp
    Q[..., 1, 2] = -ct * sp
    Q[..., 2, 2] = st

    return Q


def block_rotations(Q: np.ndarray) -> np.ndarray:
    """
    Embed each 3x3 rotation as the 6x6 block-diagonal ``blockdiag(Q, Q)``.

    The 6-component Cosserat state stacks a displacement 3-vector and a microrotation
    3-vector; both transform by the same rotation ``Q``, so the state transforms by
    ``D = blockdiag(Q, Q)`` with zero off-diagonal blocks.

    Parameters
    ----------
    Q : np.ndarray
        Shape (..., 3, 3) rotations.

    Returns
    -------
    np.ndarray
        Shape (..., 6, 6), float64, with ``Q`` in both diagonal 3x3 blocks and zeros
        off-diagonal.

    Raises
    ------
    ValueError
        If the trailing two dimensions of ``Q`` are not (3, 3).
    """
    Q = np.asarray(Q, dtype=float)
    if Q.shape[-2:] != (3, 3):
        err = f"Q must have trailing shape (3, 3), got {Q.shape}."
        logger.error(err)
        raise ValueError(err)

    D = np.zeros((*Q.shape[:-2], 6, 6), dtype=np.float64)
    D[..., :3, :3] = Q
    D[..., 3:, 3:] = Q
    return D


def generate_spherical_grid(
    radii: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Build the direction unit vectors and Cartesian coordinates of a spherical grid.

    Parameters
    ----------
    radii : np.ndarray
        1-D strictly-positive radii, shape (n_radii,).
    theta : np.ndarray
        1-D polar angles from +z in [0, pi], shape (n_theta,).
    phi : np.ndarray
        1-D azimuths from +x toward +y in [0, 2*pi), shape (n_phi,).

    Returns
    -------
    directions : np.ndarray
        Shape (n_theta, n_phi, 3) unit vectors ``r_hat(theta, phi)``.
    coordinates : np.ndarray
        Shape (n_radii, n_theta, n_phi, 3), equal to
        ``radii[:, None, None, None] * directions`` with
        ``x = r sin(theta) cos(phi)``, ``y = r sin(theta) sin(phi)``,
        ``z = r cos(theta)``.

    Raises
    ------
    ValueError
        If any input is not 1-D, any radius is non-positive, ``theta`` is outside
        [0, pi], or ``phi`` is outside [0, 2*pi).
    """
    radii = np.asarray(radii, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    for name, arr in (("radii", radii), ("theta", theta), ("phi", phi)):
        if arr.ndim != 1:
            err = f"{name} must be a 1D array."
            logger.error(err)
            raise ValueError(err)
    if np.any(radii <= 0.0):
        err = "radii must be strictly positive (the source sits at the origin)."
        logger.error(err)
        raise ValueError(err)
    if np.any(theta < 0.0) or np.any(theta > np.pi):
        err = "theta must lie in [0, pi]."
        logger.error(err)
        raise ValueError(err)
    if np.any(phi < 0.0) or np.any(phi >= 2.0 * np.pi):
        err = "phi must lie in [0, 2*pi)."
        logger.error(err)
        raise ValueError(err)

    st = np.sin(theta)[:, None]
    ct = np.cos(theta)[:, None]
    cp = np.cos(phi)[None, :]
    sp = np.sin(phi)[None, :]

    directions = np.empty((theta.shape[0], phi.shape[0], 3), dtype=np.float64)
    directions[..., 0] = st * cp
    directions[..., 1] = st * sp
    directions[..., 2] = np.broadcast_to(ct, (theta.shape[0], phi.shape[0]))

    coordinates = radii[:, None, None, None] * directions[None, ...]
    return directions, coordinates


def rotate_and_contract(
    radial_snapshot: np.ndarray,
    D: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    r"""
    Reconstruct the field ``v = D G D^T f`` for one time slice, over all directions.

    By isotropic rotational covariance ``G(Q x, omega) = D G(x, omega) D^T`` (proper
    rotations, ``D = blockdiag(Q, Q)``), the field at direction ``d`` and radius ``r``
    is ``v = D[d] @ G_radial(r) @ D[d]^T @ f`` with the source vector ``f`` fixed in
    space. The rotated 6x6 tensor field is never materialized (that would be
    O(n_radii * n_dir * 36)); only the 6-vectors ``D^T f`` and the field itself are.

    Parameters
    ----------
    radial_snapshot : np.ndarray
        Shape (n_radii, 6, 6): the radial Green tensor at one time slice.
    D : np.ndarray
        Shape (n_dir, 6, 6): block rotations for each flattened direction.
    f : np.ndarray
        Shape (6,): the space-fixed source vector (``source.direction()``).

    Returns
    -------
    np.ndarray
        Shape (n_radii, n_dir, 6), float64: the reconstructed 6-component field.
    """
    # D^T f: (n_dir, 6). Cheap (n_dir * 36) and independent of n_radii.
    w = np.einsum("dcb,c->db", D, f)
    # G @ (D^T f): (n_radii, n_dir, 6).
    Gw = np.einsum("rbc,dc->rdb", radial_snapshot, w)
    # D @ (G D^T f): (n_radii, n_dir, 6).
    return np.einsum("dab,rdb->rda", D, Gw)


COMPONENT_NAMES = ["ux", "uy", "uz", "rx", "ry", "rz"]


def write_spherical_wavefield(
    output_path: str,
    radii: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    dim: int,
    material_type: int,
    material_params: dict,
    sources: list[SourceSpectrum],
    simulation_params: dict,
    use_fortran: bool = True,
    force_use_openmp: bool = False,
    force_no_openmp: bool = False,
    radial_chunk_size: int | None = None,
    compression: str | None = "gzip",
    write_xdmf: bool = True,
    close_phi_seam: bool = True,
) -> dict:
    """
    Compute and write the full spherical wavefield to HDF5 (+ XDMF sidecar).

    Exploits isotropic rotational covariance: the radial Green tensor is computed once
    per source along +x, then the field at every ``(theta, phi)`` direction is
    reconstructed by rotation. Sources are processed sequentially with an HDF5
    read-accumulate-write so peak memory never grows with the number of sources; the
    largest retained buffer is a single snapshot ``(n_r, n_theta, n_phi, 6)``.

    Layout (SPECFEM-like): one group per snapshot, named ``Step<index>`` by the
    sample index into the N-grid, each holding two named vector datasets
    ``Displacement`` and ``Rotation`` of shape ``(n_r, n_theta, n_phi, 3)``. The
    curvilinear point coordinates and snapshot times are stored once at the root
    (``/coordinates``, ``/times``).

    Parameters
    ----------
    output_path : str
        Path to the ``.h5`` file. The XDMF sidecar is written beside it with the same
        stem and a ``.xdmf`` suffix.
    radii : np.ndarray
        1-D strictly-positive radii.
    theta : np.ndarray
        1-D polar angles from +z in [0, pi].
    phi : np.ndarray
        1-D azimuths from +x toward +y in [0, 2*pi).
    dim : int
        Only ``consts.DIMENSION_3D`` is supported.
    material_type : int
        Cosserat or elastic.
    material_params : dict
        Full material parameter dict (all 8 Cosserat keys; dummies ignored for
        elastic).
    sources : list[SourceSpectrum]
        Sources at the origin. Their contributions are summed.
    simulation_params : dict
        Fourier parameters. Not mutated; a copy is taken and ``t0`` resolved via
        :func:`_resolve_t0` for the shared grid.
    use_fortran : bool, default=True
        Must be True; spherical wavefields require the Fortran backend.
    force_use_openmp, force_no_openmp : bool
        OpenMP overrides passed through to the backend.
    radial_chunk_size : int | None, default=None
        Radial chunking for the backend/transform workspace (see
        :func:`compute_source_radial_tensor`).
    compression : str | None, default="gzip"
        HDF5 compression filter for ``/values`` (None disables it).
    write_xdmf : bool, default=True
        If True, write an XDMF sidecar for ParaView.
    close_phi_seam : bool, default=True
        If True, append a duplicate of the ``phi=0`` slice at ``phi=2*pi`` to both
        the coordinates and every field, closing the periodic azimuthal seam so the
        rendered mesh is watertight (no missing wedge between the last ``phi`` column
        and ``phi=0``). Only correct when ``phi`` spans the full circle; set False for
        a partial-``phi`` (wedge) grid. Adds one column to the written ``phi`` axis.

    Returns
    -------
    dict
        Metadata: ``h5_path``, ``xdmf_path`` (or None), ``times``, ``dt``,
        ``steps_per_snapshot``, ``n_snapshots``, ``step_names``, ``field_shape``
        (the per-step ``Displacement``/``Rotation`` shape), and
        ``coordinates_shape``.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    ValueError
        Propagated from the validation in the delegated functions.
    """
    try:
        import h5py  # noqa: PLC0415  since we don't want to import unless using sphericalwavefield
    except ImportError as exc:
        err = (
            "h5py is required for wavefield output; "
            "install with `pip install cosserat_solver[wavefield]`."
        )
        logger.error(err)
        raise ImportError(err) from exc

    if not sources:
        err = "sources must be a non-empty list."
        logger.error(err)
        raise ValueError(err)

    # Shared time grid: copy params (do not mutate the caller's dict) and resolve t0.
    sim = dict(simulation_params)
    sim["t0"] = _resolve_t0(sim, sources)

    # Grid geometry and rotations (validation happens inside these helpers).
    _directions, coordinates = generate_spherical_grid(radii, theta, phi)
    radii = np.asarray(radii, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    n_r, n_theta, n_phi = radii.shape[0], theta.shape[0], phi.shape[0]
    n_dir = n_theta * n_phi

    snapshot_idx = get_snapshot_indices(sim)
    n_snap = snapshot_idx.shape[0]
    interval = int(sim.get("steps_per_snapshot", 1))

    Q = rotation_matrices(theta, phi).reshape(n_dir, 3, 3)
    D = block_rotations(Q)  # (n_dir, 6, 6)

    full_times = fourier._time_array(sim)
    snapshot_times = full_times[snapshot_idx]

    # Close the periodic azimuthal seam: append the phi=0 slice at phi=2*pi so the
    # curvilinear mesh has no missing wedge. Coordinates are wrapped once here; each
    # reconstructed field is wrapped the same way as it is written.
    close_seam = close_phi_seam and n_phi >= 2
    if close_seam:
        phi_written = np.concatenate([phi, [2.0 * np.pi]])
        coordinates = np.concatenate([coordinates, coordinates[:, :, :1, :]], axis=2)
    else:
        phi_written = phi
    n_phi_w = phi_written.shape[0]

    # SPECFEM-like layout: one group per snapshot, each holding named vector
    # datasets Displacement and Rotation. Steps are named by their sample index
    # into the N-grid (Step000000, Step000008, ...).
    step_names = [f"Step{int(idx):06d}" for idx in snapshot_idx]
    field_shape = (n_r, n_theta, n_phi_w, 3)

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("times", data=snapshot_times)
        h5.create_dataset("radii", data=radii)
        h5.create_dataset("theta", data=theta)
        h5.create_dataset("phi", data=phi_written)
        h5.create_dataset("coordinates", data=coordinates)

        h5.attrs["coordinate_system"] = "spherical"
        h5.attrs["theta_definition"] = "polar angle from +z"
        h5.attrs["phi_definition"] = "azimuth from +x toward +y"
        h5.attrs["displacement_components"] = COMPONENT_NAMES[:3]
        h5.attrs["rotation_components"] = COMPONENT_NAMES[3:]
        h5.attrs["source_center"] = np.array([0.0, 0.0, 0.0])
        h5.attrs["steps_per_snapshot"] = interval
        h5.attrs["dt"] = float(sim["dt"])
        h5.attrs["t0"] = float(sim["t0"])
        h5.attrs["material_type"] = int(material_type)

        for s_index, source in enumerate(sources):
            # All radial work for this source completes before any reconstruction.
            _, tensor = compute_source_radial_tensor(
                radii,
                sim,
                material_params,
                source,
                material_type,
                dim=dim,
                use_fortran=use_fortran,
                force_use_openmp=force_use_openmp,
                force_no_openmp=force_no_openmp,
                time_indices=snapshot_idx,
                radial_chunk_size=radial_chunk_size,
            )  # (n_r, n_snap, 6, 6)

            f = np.asarray(source.direction(), dtype=float)

            for k, step_name in enumerate(step_names):
                # (n_r, n_dir, 6) -> (n_r, n_theta, n_phi, 6), split into fields.
                v = rotate_and_contract(tensor[:, k], D, f).reshape(
                    n_r, n_theta, n_phi, 6
                )
                if close_seam:
                    # Match the wrapped coordinates: duplicate the phi=0 slice.
                    v = np.concatenate([v, v[:, :, :1, :]], axis=2)
                disp = v[..., :3]
                rot = v[..., 3:]
                if s_index == 0:
                    grp = h5.create_group(step_name)
                    grp.attrs["time"] = float(snapshot_times[k])
                    grp.attrs["step_index"] = int(snapshot_idx[k])
                    grp.create_dataset(
                        "Displacement",
                        data=disp,
                        chunks=field_shape,
                        compression=compression,
                    )
                    grp.create_dataset(
                        "Rotation",
                        data=rot,
                        chunks=field_shape,
                        compression=compression,
                    )
                else:
                    # Read-modify-write one snapshot: RAM stays O(one snapshot).
                    grp = h5[step_name]
                    grp["Displacement"][...] = grp["Displacement"][...] + disp
                    grp["Rotation"][...] = grp["Rotation"][...] + rot

            del tensor

    xdmf_path = None
    if write_xdmf:
        xdmf_path = os.path.splitext(output_path)[0] + ".xdmf"
        _write_xdmf_sidecar(
            output_path, xdmf_path, snapshot_times, step_names, n_r, n_theta, n_phi_w
        )

    return {
        "h5_path": output_path,
        "xdmf_path": xdmf_path,
        "times": snapshot_times,
        "dt": float(sim["dt"]),
        "steps_per_snapshot": interval,
        "n_snapshots": n_snap,
        "step_names": step_names,
        "field_shape": field_shape,
        "coordinates_shape": coordinates.shape,
    }


def _write_xdmf_sidecar(
    h5_path: str,
    xdmf_path: str,
    times: np.ndarray,
    step_names: list[str],
    n_r: int,
    n_theta: int,
    n_phi: int,
) -> None:
    """
    Write an XDMF sidecar describing the HDF5 wavefield for ParaView.

    A temporal ``Grid`` collection: each timestep is a curvilinear (``3DSMesh``) grid
    over the ``(n_r, n_theta, n_phi)`` coordinates, with two 3-vector attributes
    ``Displacement`` and ``Rotation`` pointing directly at that step's
    ``Step<index>/Displacement`` and ``Step<index>/Rotation`` datasets.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file. Only its basename is referenced from the XDMF so the
        pair stays relocatable together.
    xdmf_path : str
        Output path for the ``.xdmf`` file.
    times : np.ndarray
        Snapshot times, shape (n_snap,).
    step_names : list[str]
        Per-snapshot group names (e.g. ``["Step000000", "Step000008", ...]``).
    n_r, n_theta, n_phi : int
        Grid dimensions.
    """

    h5_name = os.path.basename(h5_path)
    npoints = n_r * n_theta * n_phi
    dims = f"{n_r} {n_theta} {n_phi}"

    lines = [
        '<?xml version="1.0" ?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="2.0">',
        "  <Domain>",
        '    <Grid Name="Wavefield" GridType="Collection" CollectionType="Temporal">',
    ]

    for k, step_name in enumerate(step_names):
        t = float(times[k])
        lines += [
            f'      <Grid Name="{step_name}" GridType="Uniform">',
            f'        <Time Value="{t!r}"/>',
            f'        <Topology TopologyType="3DSMesh" Dimensions="{dims}"/>',
            '        <Geometry GeometryType="XYZ">',
            f'          <DataItem Dimensions="{npoints} 3" NumberType="Float" '
            f'Precision="8" Format="HDF">{h5_name}:/coordinates</DataItem>',
            "        </Geometry>",
        ]
        # Two 3-vector attributes, each a direct reference to this step's dataset.
        for name in ("Displacement", "Rotation"):
            lines += [
                f'        <Attribute Name="{name}" AttributeType="Vector" '
                'Center="Node">',
                f'          <DataItem Dimensions="{npoints} 3" NumberType="Float" '
                f'Precision="8" Format="HDF">{h5_name}:/{step_name}/{name}</DataItem>',
                "        </Attribute>",
            ]
        lines.append("      </Grid>")

    lines += [
        "    </Grid>",
        "  </Domain>",
        "</Xdmf>",
    ]

    with open(xdmf_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
