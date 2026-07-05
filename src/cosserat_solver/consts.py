from __future__ import annotations

PLUS_BRANCH = -1

COMPUTE_PRECISION = 30  # enough to accommodate e10 differences squared
TEST_PRECISION = 33  # use more precision for testing

# material type tags
MATERIAL_TYPE_ELASTIC = 0
MATERIAL_TYPE_COSSERAT = 1

# source type tags
SOURCE_TYPE_RICKER = 0

# seismogram type tags
SEISMOGRAM_TYPE_DISPLACEMENT = 0
SEISMOGRAM_TYPE_ROTATION = 1

# dimension tags
DIMENSION_2D = 2
DIMENSION_3D = 3

# backend tags
BACKEND_AUTO = 0
BACKEND_FORTRAN = 1
BACKEND_PYTHON = 2
