cmake_minimum_required(VERSION 3.14)
project(DynamicAgents)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# FetchContent for dependencies
include(FetchContent)

# Fetch nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Fetch CPR for HTTP requests
FetchContent_Declare(
    cpr
    GIT_REPOSITORY https://github.com/libcpr/cpr.git
    GIT_TAG 1.10.5
)
FetchContent_MakeAvailable(cpr)

# Add the executable
add_executable(DynamicAgents 
    main.cpp
)

# Link libraries
target_link_libraries(DynamicAgents PRIVATE 
    nlohmann_json::nlohmann_json
    cpr::cpr
)