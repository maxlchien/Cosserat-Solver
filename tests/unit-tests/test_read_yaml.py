"""
Tests for read_yaml's spherical_wavefield block parsing.

These are pure-Python (no backend): they exercise the schema validation and
default resolution in ``_parse_spherical_wavefield`` and the cross-field checks
in ``read`` (3D-only, sources-at-center). The wavefield dict is the 9th element
(index 8) of the tuple returned by ``read``.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from cosserat_solver.read_yaml import read

WAVEFIELD_INDEX = 8


def base_config() -> dict:
    """A minimal, valid 3D Cosserat config with one origin source."""
    return {
        "dimension": 3,
        "material_type": "cosserat",
        "material_params": {
            "rho": 2300,
            "lam": 7.682e9,
            "mu": 5.175e9,
            "nu": 1e2,
            "J": 2300,
            "lam_c": 6e9,
            "mu_c": 2e9,
            "nu_c": 0,
        },
        "sources": [
            {
                "type": "Ricker",
                "location": [0.0, 0.0, 0.0],
                "f0": 0.1,
                "f": [1.0, 1.0, 1.0],
                "fc": [1.0, 1.0, 1.0],
            }
        ],
        "simulation_params": {
            "dt": 0.01,
            "N": 100,
            "refinement_factor": 1,
            "extension_factor": 2,
        },
    }


def full_wavefield_block() -> dict:
    """A wavefield block that sets every key explicitly."""
    return {
        "center": [0, 0, 0],
        "r_min": 250.0,
        "r_max": 10000.0,
        "num_radii": 40,
        "num_theta": 30,
        "num_phi": 40,
        "steps_per_snapshot": 100,
        "directory": "some/output/dir",
        "compression": "gzip",
        "write_xdmf": True,
        "radial_chunk_size": 10,
    }


def minimal_wavefield_block() -> dict:
    """Only the required keys; everything else should default."""
    return {
        "r_max": 10000.0,
        "num_radii": 40,
        "num_theta": 30,
        "num_phi": 40,
        "steps_per_snapshot": 100,
    }


def write_and_read(tmp_path, config: dict):
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(config))
    return read(str(path))


def read_wavefield(tmp_path, config: dict):
    return write_and_read(tmp_path, config)[WAVEFIELD_INDEX]


# ---------------------------------------------------------------------------
# absence + full parse
# ---------------------------------------------------------------------------


def test_wavefield_absent_is_none(tmp_path):
    assert read_wavefield(tmp_path, base_config()) is None


def test_full_block_parses(tmp_path):
    config = base_config()
    config["spherical_wavefield"] = full_wavefield_block()
    wf = read_wavefield(tmp_path, config)

    assert wf is not None
    np.testing.assert_array_equal(wf["center"], np.zeros(3))
    assert wf["r_min"] == 250.0
    assert wf["r_max"] == 10000.0
    assert wf["num_radii"] == 40
    assert wf["num_theta"] == 30
    assert wf["num_phi"] == 40
    assert wf["steps_per_snapshot"] == 100
    assert wf["directory"] == "some/output/dir"
    assert wf["compression"] == "gzip"
    assert wf["write_xdmf"] is True
    assert wf["radial_chunk_size"] == 10


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_defaults_applied_with_only_required_keys(tmp_path):
    config = base_config()
    config["spherical_wavefield"] = minimal_wavefield_block()
    wf = read_wavefield(tmp_path, config)

    # r_min defaults to r_max / num_radii (radii dr, 2dr, ..., r_max).
    assert wf["r_min"] == pytest.approx(10000.0 / 40)
    np.testing.assert_array_equal(wf["center"], np.zeros(3))
    assert wf["directory"] is None
    assert wf["compression"] == "gzip"
    assert wf["write_xdmf"] is True
    assert wf["radial_chunk_size"] is None


@pytest.mark.parametrize("value", [None, "none", "None"])
def test_compression_none_disables(tmp_path, value):
    config = base_config()
    block = minimal_wavefield_block()
    block["compression"] = value
    config["spherical_wavefield"] = block
    wf = read_wavefield(tmp_path, config)
    assert wf["compression"] is None


def test_nonorigin_center_with_matching_source_parses(tmp_path):
    config = base_config()
    config["sources"][0]["location"] = [100.0, -50.0, 25.0]
    block = minimal_wavefield_block()
    block["center"] = [100.0, -50.0, 25.0]
    config["spherical_wavefield"] = block
    wf = read_wavefield(tmp_path, config)
    np.testing.assert_array_equal(wf["center"], np.array([100.0, -50.0, 25.0]))


# ---------------------------------------------------------------------------
# missing required keys -> KeyError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    ["r_max", "num_radii", "num_theta", "num_phi", "steps_per_snapshot"],
)
def test_missing_required_key_raises(tmp_path, missing):
    config = base_config()
    block = minimal_wavefield_block()
    del block[missing]
    config["spherical_wavefield"] = block
    with pytest.raises(KeyError):
        write_and_read(tmp_path, config)


# ---------------------------------------------------------------------------
# bad values -> ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("r_max", 0),
        ("r_max", -1.0),
        ("num_radii", 0),
        ("num_theta", 1),
        ("num_phi", 0),
        ("steps_per_snapshot", 0),
        ("steps_per_snapshot", 1.5),
        ("steps_per_snapshot", True),
        ("num_radii", 2.5),
        ("write_xdmf", "yes"),
        ("radial_chunk_size", 0),
        ("directory", ""),
        ("center", [0, 0]),
        ("compression", 5),
    ],
)
def test_bad_value_raises(tmp_path, key, value):
    config = base_config()
    block = minimal_wavefield_block()
    block[key] = value
    config["spherical_wavefield"] = block
    with pytest.raises(ValueError):  # noqa: PT011  no consistent regex to match, different errors have different messages
        write_and_read(tmp_path, config)


def test_r_min_not_less_than_r_max_raises(tmp_path):
    config = base_config()
    block = minimal_wavefield_block()
    block["r_min"] = 10000.0  # == r_max
    config["spherical_wavefield"] = block
    with pytest.raises(ValueError, match=r"r_min' .* must be < 'r_max'"):
        write_and_read(tmp_path, config)


def test_r_min_zero_raises(tmp_path):
    config = base_config()
    block = minimal_wavefield_block()
    block["r_min"] = 0
    config["spherical_wavefield"] = block
    with pytest.raises(ValueError, match=r"r_min' .* must be > 0"):
        write_and_read(tmp_path, config)


# ---------------------------------------------------------------------------
# cross-field validation
# ---------------------------------------------------------------------------


def test_wavefield_in_2d_raises(tmp_path):
    config = base_config()
    config["dimension"] = 2
    # 2D sources/material still need to parse; give a 2D-shaped source location.
    config["sources"][0]["location"] = [0.0, 0.0]
    config["sources"][0]["f"] = 1.0
    config["sources"][0]["fc"] = 1.0
    config["spherical_wavefield"] = minimal_wavefield_block()
    with pytest.raises(ValueError, match="3D only"):
        write_and_read(tmp_path, config)


def test_source_off_origin_center_raises(tmp_path):
    """Default (origin) center with an off-origin source is rejected."""
    config = base_config()
    config["sources"][0]["location"] = [1.0, 0.0, 0.0]
    config["spherical_wavefield"] = minimal_wavefield_block()
    with pytest.raises(ValueError, match="center"):
        write_and_read(tmp_path, config)


def test_source_not_matching_nonorigin_center_raises(tmp_path):
    """Non-origin center with a source at the origin is rejected."""
    config = base_config()
    block = minimal_wavefield_block()
    block["center"] = [100.0, 0.0, 0.0]
    config["spherical_wavefield"] = block
    with pytest.raises(ValueError, match="center"):
        write_and_read(tmp_path, config)


# ---------------------------------------------------------------------------
# extras
# ---------------------------------------------------------------------------


def test_extra_keys_are_ignored(tmp_path):
    config = base_config()
    block = minimal_wavefield_block()
    block["not_a_real_key"] = 123
    config["spherical_wavefield"] = block
    wf = read_wavefield(tmp_path, config)
    assert "not_a_real_key" not in wf
