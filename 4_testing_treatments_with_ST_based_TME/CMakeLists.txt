### CMakeLists for MSABM with optimalRL treatment test on ST based data
project(optimalRL_treatment_ST) # CSF1RIonly_treatment_ST, IGF1RIonly_treatment_ST or CSF1RIandIGF1RI_treatment_ST
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin_ST)
aux_source_directory(src SRC_LIST)
include_directories(inc)
add_executable(optimalRL_treatment_ST ${SRC_LIST})