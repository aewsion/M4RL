### CMakeLists for MSABM tumor_growth_case
project(tumor_growth_case)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(src SRC_LIST)
include_directories(inc)
add_executable(tumor_growth_case ${SRC_LIST})