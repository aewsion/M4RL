### CMakeLists for MSABM no treatment case
project(no_treatment_case)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(src SRC_LIST)
include_directories(inc)
add_executable(no_treatment_case ${SRC_LIST})