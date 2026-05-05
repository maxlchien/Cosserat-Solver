# --- branch_config generation (unchanged) ---
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dim2)
# TODO: these should both be removed once dim2 is reworked
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dim2/branch_config.f90
  COMMAND ${Python_EXECUTABLE}

    ${CMAKE_CURRENT_SOURCE_DIR}/dim2/generate_consts.py
    ${CMAKE_CURRENT_SOURCE_DIR}/../consts.py
    ${CMAKE_CURRENT_BINARY_DIR}/dim2/branch_config.f90
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/../consts.py
    ${CMAKE_CURRENT_SOURCE_DIR}/dim2/generate_consts.py
  VERBATIM
)

# --- Build dispersion_fortran library ---
add_library(dispersion_fortran STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/kinds.f90
  ${CMAKE_CURRENT_BINARY_DIR}/dim2/branch_config.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/dispersion_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/dispersion_core_c.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/integrator_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/integrator_core_c.f90
)

target_compile_options(dispersion_fortran PRIVATE
  -fdefault-real-8
  -fdefault-double-8
)

set_target_properties(dispersion_fortran PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# --- AMOS/SLATEC integration in subfolder fortran/amos ---
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/amos/amos_deps.txt" AMOS_SOURCES)
# prepend folder path to each source
list(TRANSFORM AMOS_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/amos/")

add_library(amos STATIC ${AMOS_SOURCES})

# DO NOT apply -fdefault-real-8 to AMOS library!
# AMOS is F77 code that expects standard double precision
# and uses machine constants that break with promotion flags
set_target_properties(amos PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# --- Build Python extension that exposes dispersion_core ---
Python_add_library(dispersion_core MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/dispersion_wrapper.c
)

# --- Build Python extension that exposes integrator_core ---
Python_add_library(integrator_core MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/integrator_wrapper.c
)

target_link_libraries(dispersion_core PRIVATE
  dispersion_fortran
)

target_link_libraries(integrator_core PRIVATE
  dispersion_fortran
  amos
)


install(TARGETS dispersion_core DESTINATION cosserat_solver)
install(TARGETS integrator_core DESTINATION cosserat_solver)
