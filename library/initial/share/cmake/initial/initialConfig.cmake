# Find initial
# -------
#
# Finds initial
#
# This will define the following variables:
#
#   initial_FOUND        -- True if the system has initial
#   initial_INCLUDE_DIRS -- The include directories for initial
#   initial_LIBRARIES    -- Libraries to link against
#
# and the following imported targets:
#
#   initial

# Find initial root
# Assume we are in ${initialROOT}/share/cmake/initial/initialConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(initialROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(initial_INCLUDE_DIRS ${initialROOT}/include)

# library
add_library(initial STATIC IMPORTED)
set(initial_LIBRARIES initial)

# dependency: Torch-Chemistry
if(NOT tchem_FOUND)
    find_package(tchem REQUIRED PATHS ~/Library/Torch-Chemistry)
    list(APPEND initial_INCLUDE_DIRS ${tchem_INCLUDE_DIRS})
    list(APPEND initial_LIBRARIES ${tchem_LIBRARIES})
    set(initial_CXX_FLAGS "${tchem_CXX_FLAGS}")
endif()

# import location
find_library(initial_LIBRARY initial PATHS "${initialROOT}/lib")
set_target_properties(initial PROPERTIES
    IMPORTED_LOCATION "${initial_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${initial_INCLUDE_DIRS}"
    CXX_STANDARD 14
)