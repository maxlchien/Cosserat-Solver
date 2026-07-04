# --- branch_config generation (unchanged) ---
# file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
# TODO: these should both be removed once dim2 is reworked
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/branch_config.f90
  COMMAND ${Python_EXECUTABLE}

    ${CMAKE_CURRENT_SOURCE_DIR}/generate_consts.py
    ${CMAKE_CURRENT_SOURCE_DIR}/../../consts.py
    ${CMAKE_CURRENT_BINARY_DIR}/branch_config.f90
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../consts.py
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_consts.py
  VERBATIM
)

add_library(dim2_cosserat STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/branch_config.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dispersion_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dispersion_core_c.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/integrator_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/integrator_core_c.f90
)
target_link_libraries(dim2_cosserat PUBLIC kinds)
target_compile_options(dim2_cosserat PRIVATE
  -fdefault-real-8
  -fdefault-double-8
)
set_target_properties(dim2_cosserat PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cosserat
)
# Add OpenMP flags if available
if(OpenMP_Fortran_FOUND)
  target_compile_options(dim2_cosserat PRIVATE ${OpenMP_Fortran_FLAGS})
  target_link_libraries(dim2_cosserat PRIVATE OpenMP::OpenMP_Fortran)
endif()

# --- Build Python extension that exposes dispersion_core ---
Python_add_library(dispersion_core_2d MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/dispersion_wrapper.c
)

# --- Build Python extension that exposes integrator_core ---
Python_add_library(integrator_core_2d MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/integrator_wrapper.c
)

target_link_libraries(dispersion_core_2d PRIVATE
  dim2_cosserat
)

target_link_libraries(integrator_core_2d PRIVATE
  dim2_cosserat
  amos
)



install(TARGETS dispersion_core_2d DESTINATION cosserat_solver/dim2)
install(TARGETS integrator_core_2d DESTINATION cosserat_solver/dim2)
