cmake_minimum_required(VERSION 3.15)
project(cosserat_solver LANGUAGES C Fortran)

# Find Python and NumPy
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Set Fortran flags
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fdefault-real-8 -fdefault-double-8")

# Add the package
add_subdirectory(src/cosserat_solver)
