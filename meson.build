project(
  'cosserat_solver',
  ['fortran'],
  version: '0.1.0',
  default_options: [
    'buildtype=release',
    'fortran_std=f2008'
  ]
)

# Python is needed to generate Fortran sources
py = import('python').find_installation()

message('Using Python at: ' + py.path())

# Enter the Fortran backend
subdir('src/cosserat_solver/fortran')
