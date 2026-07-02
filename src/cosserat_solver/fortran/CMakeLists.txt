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

# --- Build compiled library ---
add_library(kinds STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/kinds.f90
)

target_compile_options(kinds PRIVATE
  -fdefault-real-8
  -fdefault-double-8
)
set_target_properties(kinds PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kinds
)
target_include_directories(kinds INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/kinds)

add_library(dim2 STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/dim2/branch_config.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/dispersion_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/dispersion_core_c.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/integrator_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/integrator_core_c.f90
)
target_link_libraries(dim2 PUBLIC kinds)
target_compile_options(dim2 PRIVATE
  -fdefault-real-8
  -fdefault-double-8
)
set_target_properties(dim2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dim2
)

add_library(dim3 STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/dim3/dispersion_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim3/dispersion_core_c.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim3/cosserat_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dim3/cosserat_core_c.f90
)
target_link_libraries(dim3 PUBLIC kinds)
target_compile_options(dim3 PRIVATE
  -fdefault-real-8
  -fdefault-double-8
)
set_target_properties(dim3 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dim3
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
Python_add_library(dispersion_core_2d MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/dispersion_wrapper.c
)

# --- Build Python extension that exposes integrator_core ---
Python_add_library(integrator_core_2d MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/dim2/integrator_wrapper.c
)

# --- Build Python extension that exposes dispersion_core_3d ---
Python_add_library(dispersion_core_3d MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/dim3/dispersion_wrapper.c
)

# --- Build Python extension that exposes cosserat_core_3d ---
Python_add_library(cosserat_core_3d MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/dim3/cosserat_wrapper.c
)

target_link_libraries(dispersion_core_2d PRIVATE
  dim2
)

target_link_libraries(integrator_core_2d PRIVATE
  dim2
  amos
)

target_link_libraries(dispersion_core_3d PRIVATE
  dim3
)

target_link_libraries(cosserat_core_3d PRIVATE
  dim3
)


install(TARGETS dispersion_core_2d DESTINATION cosserat_solver)
install(TARGETS integrator_core_2d DESTINATION cosserat_solver)
install(TARGETS dispersion_core_3d DESTINATION cosserat_solver)
install(TARGETS cosserat_core_3d DESTINATION cosserat_solver)
