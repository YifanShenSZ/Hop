cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(PCGFSH)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# Cpp-Library
set(CMAKE_PREFIX_PATH ~/Library/Cpp-Library)
find_package(CL REQUIRED)

# hop
set(CMAKE_PREFIX_PATH ~/Software/Mine/hop/library/hop)
find_package(hop REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${hop_CXX_FLAGS}")

add_executable(PCGFSH.exe 
    ../common/Hd.cpp
    main.cpp
)

target_link_libraries(PCGFSH.exe
    ${hop_LIBRARIES}
    ${CL_LIBRARIES}
)