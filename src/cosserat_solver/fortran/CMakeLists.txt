# Generate branch_config.f90 from Python constants
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/branch_config.f90
  COMMAND ${Python_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_consts.py
    ${CMAKE_CURRENT_SOURCE_DIR}/../consts.py
    ${CMAKE_CURRENT_BINARY_DIR}/branch_config.f90
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/../consts.py
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_consts.py
  VERBATIM
)

# Build Fortran code as a static library
add_library(dispersion_fortran STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/kinds.f90
  ${CMAKE_CURRENT_BINARY_DIR}/branch_config.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dispersion_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dispersion_core_c.f90
)

target_compile_options(dispersion_fortran PRIVATE
  -fdefault-real-8
  -fdefault-double-8
)

set_target_properties(dispersion_fortran PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# Build Python extension that links to rmFortran library
Python_add_library(dispersion_core MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/dispersion_wrapper.c
)

target_link_libraries(dispersion_core PRIVATE
  dispersion_fortran
)

install(TARGETS dispersion_core DESTINATION cosserat_solver)
