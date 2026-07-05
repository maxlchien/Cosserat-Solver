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
    session.install("--no-build-isolation", "--force-reinstall", "-e", ".")

    # session.install(".[test]")
    session.run("pytest", "tests", *session.posargs)


@nox_uv.session(uv_groups=["test"])
def displacement_tests(session: nox.Session) -> None:
    """
    Run the displacement tests.

    If a positional argument is provided, it will be used as the maximum number of cores to be used across all tests.
    If no positional argument is provided, the default maximum number of cores will be 32.
    """
    session.install("--no-build-isolation", "--force-reinstall", "-e", ".")

    # session.install(".[test]")
    session.run(
        "pytest",
        "tests",
        "--displacement-tests",
        "--max-cores=" + str(session.posargs[0])
        if session.posargs
        else "--max-cores=32",
        *session.posargs[1:] if len(session.posargs) > 1 else [],
    )


@nox_uv.session(uv_groups=["dev"])
def comparisons(session: nox.Session) -> None:
    """
    Run the comparisons.
    """
    comparisons_dir = pathlib.Path("comparisons")

    for subfolder in comparisons_dir.iterdir():
        if not subfolder.is_dir():
            continue

        # run solver
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

        # run specfem (via snakemake)
        master_config_file = subfolder / "specfem" / "master_config.yaml"
        if master_config_file.exists():
            ...
