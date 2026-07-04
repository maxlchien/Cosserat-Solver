add_library(dim3_cosserat STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/dispersion_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/dispersion_core_c.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/cosserat_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/cosserat_core_c.f90
)
target_link_libraries(dim3_cosserat PUBLIC kinds)
target_compile_options(dim3_cosserat PRIVATE
  -fdefault-real-8
  -fdefault-double-8
)
set_target_properties(dim3_cosserat PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cosserat
)
# Add OpenMP flags if available
if(OpenMP_Fortran_FOUND)
  target_compile_options(dim3_cosserat PRIVATE ${OpenMP_Fortran_FLAGS})
  target_link_libraries(dim3_cosserat PRIVATE OpenMP::OpenMP_Fortran)
endif()

add_library(dim3_elastic STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/elastic_core.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/elastic_core_c.f90
)
target_link_libraries(dim3_elastic PUBLIC kinds)
target_compile_options(dim3_elastic PRIVATE
  -fdefault-real-8
  -fdefault-double-8
)
set_target_properties(dim3_elastic PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/elastic
)
# Add OpenMP flags if available
if(OpenMP_Fortran_FOUND)
  target_compile_options(dim3_elastic PRIVATE ${OpenMP_Fortran_FLAGS})
  target_link_libraries(dim3_elastic PRIVATE OpenMP::OpenMP_Fortran)
endif()

# --- Build Python extension that exposes dispersion_core_3d ---
Python_add_library(dispersion_core_3d MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/dispersion_wrapper.c
)

# --- Build Python extension that exposes cosserat_core_3d ---
Python_add_library(cosserat_core_3d MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/cosserat_wrapper.c
)

# --- Build Python extension that exposes elastic_core_3d ---
Python_add_library(elastic_core_3d MODULE WITH_SOABI
  ${CMAKE_CURRENT_SOURCE_DIR}/elastic_wrapper.c
)

target_link_libraries(dispersion_core_3d PRIVATE
  dim3_cosserat
)

target_link_libraries(cosserat_core_3d PRIVATE
  dim3_cosserat
)

target_link_libraries(elastic_core_3d PRIVATE
  dim3_elastic
)

# drop postfix in the python package name
set_target_properties(dispersion_core_3d PROPERTIES
  OUTPUT_NAME "dispersion_core"
)
set_target_properties(cosserat_core_3d PROPERTIES
  OUTPUT_NAME "cosserat_core"
)
set_target_properties(elastic_core_3d PROPERTIES
  OUTPUT_NAME "elastic_core"
)

install(TARGETS dispersion_core_3d DESTINATION cosserat_solver/dim3)
install(TARGETS cosserat_core_3d DESTINATION cosserat_solver/dim3)
install(TARGETS elastic_core_3d DESTINATION cosserat_solver/dim3)
