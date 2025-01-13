cmake_minimum_required(VERSION 3.14)
project(DynamicAgents)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# FetchContent for nlohmann/json
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Fetch llama.cpp
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(llama)

# Find OpenBLAS
find_package(OpenBLAS REQUIRED)

# Add the executable
add_executable(DynamicAgents 
    main.cpp
    llama_context.cpp
)

# Include directories
target_include_directories(DynamicAgents PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${llama_SOURCE_DIR}/include
    ${llama_SOURCE_DIR}/ggml/include
    ${CMAKE_BINARY_DIR}/_deps/llama-src/include
)

# Link libraries
target_link_libraries(DynamicAgents PRIVATE 
    nlohmann_json::nlohmann_json
    llama
    OpenBLAS::OpenBLAS
)