# Project-level meson.build

# Find Python for generating the Fortran config
py = import('python').find_installation()

# Path to the Python constants file
branch_py = '../consts.py'

# Path to the Python constants file
branch_py = files('../consts.py')

# Generate branch_config.f90
branch_config = custom_target(
  'branch_config',
  input: branch_py,
  output: 'branch_config.f90',
  command: [
    py, '-c',
    '''
import ast, pathlib, sys

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
    raise RuntimeError("PLUS_BRANCH not found in consts.py")

out_text = f"""module cosserat_branch_config
  use iso_fortran_env, only: int32
  implicit none
  integer(int32), parameter :: PLUS_BRANCH = {PLUS}
  integer(int32), parameter :: MINUS_BRANCH = {-PLUS}
end module cosserat_branch_config
"""

pathlib.Path(out_path).write_text(out_text)
''',
    '@INPUT@', '@OUTPUT@',
  ],
  build_by_default: true,
)

# Static library placeholder (will grow)
cosserat_fortran = static_library(
  'cosserat_fortran',
  sources: [
    'kinds.f90',
    branch_config,
  ],
)

# expose the library to parent projects
cosserat_dep = declare_dependency(
  link_with: cosserat_fortran,
)

test_branch_exe = executable(
  'test_branch',
  sources: 'test_branch.f90',
  include_directories: include_directories('build'),  # if needed for cosserat_branch_config
  link_with: cosserat_fortran,
)

test('test branch values', test_branch_exe)
