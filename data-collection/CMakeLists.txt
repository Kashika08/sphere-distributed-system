cmake_minimum_required(VERSION 3.10)
project(fetch_pod_metrics_project)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.2   
)
FetchContent_MakeAvailable(nlohmann_json)

find_package(CURL REQUIRED)

# Include directories for the project
include_directories(
    "/usr/include"
    "/usr/local/include"
    "../kafka/modern-cpp-kafka/include/"
)

link_directories(
    "/usr/lib"
    "/usr/local/lib"
)

# Building the producer as a library
add_library(producer ../kafka/Producer/producer.cpp ../kafka/Producer/producer.h)

# Constants library, assuming it is used across multiple files
add_library(constants ../kafka/Constants/constants.cpp ../kafka/Constants/constants.h)

# Executable for fetching and streaming pod metrics
add_executable(fetch_pod_metrics fetch_pod_metrics.cpp)

# Linking the fetch_pod_metrics executable with producer_lib and other dependencies
target_link_libraries(fetch_pod_metrics
    PRIVATE
    producer
    constants
    nlohmann_json::nlohmann_json
    rdkafka++  # C++ wrapper for Kafka (MUST come first)
    rdkafka    # Core C library for Kafka
    pthread ssl crypto z curl  # Required dependencies for system and Kafka
    CURL::libcurl
)
