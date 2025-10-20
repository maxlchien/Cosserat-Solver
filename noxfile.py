from __future__ import annotations

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
