from __future__ import annotations

import pathlib

import nox
import nox_uv

nox.options.default_venv_backend = "uv"


@nox_uv.session(uv_groups=["test"])
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    # session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox_uv.session(uv_groups=["dev"])
def comparisons(session: nox.Session) -> None:
    """
    Run the comparisons.
    """
    comparisons_dir = pathlib.Path("comparisons")

    for subfolder in comparisons_dir.iterdir():
        if not subfolder.is_dir():
            continue

        params_file = subfolder / "solver" / "params.yaml"
        if params_file.exists():
            output_dir = subfolder / "solver" / "output"
            session.run(
                "uv",
                "run",
                "cosserat-solver",
                "--yaml",
                str(params_file),
                "--o",
                str(output_dir),
                external=True,
            )
            session.run("python", "plot_seismograms.py", str(output_dir))
