from __future__ import annotations

import ast
import pathlib
import sys

src_path = sys.argv[1]
out_path = sys.argv[2]

src_text = pathlib.Path(src_path).read_text()
tree = ast.parse(src_text)
PLUS = None

for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if getattr(t, "id", None) == "PLUS_BRANCH":
                PLUS = ast.literal_eval(node.value)

if PLUS is None:
    err = "PLUS_BRANCH not found in consts.py"
    raise RuntimeError(err)

out_text = f"""module cosserat_branch_consts
  use iso_fortran_env, only: int32
  implicit none
  integer(int32), parameter :: PLUS_BRANCH = {PLUS}
  integer(int32), parameter :: MINUS_BRANCH = {-PLUS}
end module cosserat_branch_consts
"""

pathlib.Path(out_path).write_text(out_text)
