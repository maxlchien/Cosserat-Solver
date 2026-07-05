# --- Make kind information public across dimensions ---
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
