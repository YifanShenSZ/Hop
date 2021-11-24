cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# Torch-Chemistry
set(CMAKE_PREFIX_PATH ~/Library/Torch-Chemistry)
find_package(tchem REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TCHEM_CXX_FLAGS}")

include_directories(include)

add_library(hop STATIC
    source/base.cpp
    source/FSSH.cpp
    source/PCFSSH.cpp
    source/GFSH.cpp
    source/PCGFSH.cpp
)

target_link_libraries(hop "${tchem_LIBRARIES}")