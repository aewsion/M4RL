### CMakeLists for MSABM continuous_CSF1R_I_treatment_case
project(continuous_CSF1R_I_treatment_case)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(src SRC_LIST)
include_directories(inc)
add_executable(continuous_CSF1R_I_treatment_case ${SRC_LIST})