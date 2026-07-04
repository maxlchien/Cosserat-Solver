# --- Find OpenMP ---
find_package(OpenMP)
if(OpenMP_Fortran_FOUND)
  message(STATUS "OpenMP found for Fortran: ${OpenMP_Fortran_FLAGS}")
else()
  message(WARNING "OpenMP not found. Vectorized operations will run serially.")
endif()


# Add the subdirectories
add_subdirectory(fortran_common)
add_subdirectory(dim2/fortran)
add_subdirectory(dim3/fortran)
