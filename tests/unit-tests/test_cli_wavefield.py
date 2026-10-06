"""
End-to-end CLI test for the spherical_wavefield block.

Drives ``cli.main`` with a monkeypatched ``sys.argv`` on a tiny config and
asserts the HDF5 + XDMF outputs land where expected with the right snapshot
groups. Gated on h5py and the Fortran backend (the wavefield path requires
both). Also checks the fail-fast negative path (Python backend + wavefield).
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

import cosserat_solver.cli as cli
from cosserat_solver.greens_wrapper import FORTRAN_AVAILABLE

pytest_fortran = pytest.mark.skipif(
    not FORTRAN_AVAILABLE, reason="Fortran backend not available"
)


def tiny_config() -> dict:
    """A small 3D Cosserat config with one origin source and no receivers."""
    return {
        "dimension": 3,
        "material_type": "cosserat",
        "material_params": {
            "rho": 1.0e3,
            "lam": 1.0e5,
            "mu": 1.0e5,
            "nu": 1.0e4,
            "J": 1.0,
            "lam_c": 1.0e5,
            "mu_c": 1.0e5,
            "nu_c": 1.0e4,
        },
        "sources": [
            {
                "type": "Ricker",
                "location": [0.0, 0.0, 0.0],
                "f0": 10.0,
                "f": [1.0, -0.5, 0.3],
                "fc": [0.2, 0.7, -0.4],
            }
        ],
        "simulation_params": {
            "dt": 0.002,
            "N": 32,
            "refinement_factor": 1,
            "extension_factor": 2,
        },
        "spherical_wavefield": {
            # r_min omitted -> defaults to r_max / num_radii.
            "r_max": 60.0,
            "num_radii": 3,
            "num_theta": 4,
            "num_phi": 4,
            "steps_per_snapshot": 8,  # N=32 -> Step 0, 8, 16, 24
        },
    }


def write_config(tmp_path, config: dict) -> str:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


@pytest_fortran
def test_cli_writes_wavefield(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")

    yaml_path = write_config(tmp_path, tiny_config())
    out_dir = tmp_path / "OUTPUT_FILES"

    monkeypatch.setattr(
        "sys.argv",
        ["cosserat-solver", "--yaml", yaml_path, "--o", str(out_dir)],
    )
    cli.main()

    wf_dir = out_dir / "spherical_wavefield"
    h5_path = wf_dir / "spherical_wavefield.h5"
    xdmf_path = wf_dir / "spherical_wavefield.xdmf"
    assert h5_path.exists()
    assert xdmf_path.exists()

    with h5py.File(h5_path, "r") as h5:
        step_groups = sorted(k for k in h5 if k.startswith("Step"))
        assert step_groups == [
            "Step000000",
            "Step000008",
            "Step000016",
            "Step000024",
        ]
        # r_min defaulted to r_max / num_radii = 20 -> radii 20, 40, 60.
        np.testing.assert_allclose(h5["radii"][:], [20.0, 40.0, 60.0])
        # phi seam is closed for ParaView: num_phi + 1 columns written.
        first = h5[step_groups[0]]["Displacement"]
        assert first.shape == (3, 4, 5, 3)
        np.testing.assert_allclose(h5.attrs["source_center"], np.zeros(3))


@pytest_fortran
def test_cli_wavefield_python_backend_exits(tmp_path, monkeypatch):
    pytest.importorskip("h5py")

    yaml_path = write_config(tmp_path, tiny_config())
    out_dir = tmp_path / "OUTPUT_FILES"

    monkeypatch.setattr(
        "sys.argv",
        [
            "cosserat-solver",
            "--yaml",
            yaml_path,
            "--o",
            str(out_dir),
            "--use-python-backend",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
