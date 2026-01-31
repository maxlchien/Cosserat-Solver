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
    session.run("pytest", *session.posargs)


@nox_uv.session(uv_groups=["dev"])
def comparisons(session: nox.Session) -> None:
    """
    Run the comparisons.
    """
    session.install("--no-build-isolation", "--force-reinstall", "-e", ".")

    comparisons_dir = pathlib.Path("comparisons")

    # Determine which folders to process
    if session.posargs:
        # Use specified folders
        subfolders_to_process = []
        for folder_name in session.posargs:
            folder_path = comparisons_dir / folder_name
            if not folder_path.exists() or not folder_path.is_dir():
                err = f"Comparison folder '{folder_name}' does not exist in {comparisons_dir}"
                raise ValueError(err)
            subfolders_to_process.append(folder_path)
    else:
        # Use all folders
        subfolders_to_process = [f for f in comparisons_dir.iterdir() if f.is_dir()]

    for subfolder in subfolders_to_process:
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

        # run comparison plot
        folder_name = subfolder.name  # strip the comparisons/ part
        session.run(
            "uv",
            "run",
            "compare_seismograms.py",
            "--folder",
            folder_name,
            "--plot",
            "--all-stations",
            external=True,
        )
