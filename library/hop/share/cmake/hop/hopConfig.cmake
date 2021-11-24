# Find hop
# -------
#
# Finds hop
#
# This will define the following variables:
#
#   hop_FOUND        -- True if the system has hop
#   hop_INCLUDE_DIRS -- The include directories for hop
#   hop_LIBRARIES    -- Libraries to link against
#
# and the following imported targets:
#
#   hop

# Find hop root
# Assume we are in ${hopROOT}/share/cmake/hop/hopConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(hopROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(hop_INCLUDE_DIRS ${hopROOT}/include)

# library
add_library(hop STATIC IMPORTED)
set(hop_LIBRARIES hop)

# dependency: Torch-Chemistry
if(NOT tchem_FOUND)
    find_package(tchem REQUIRED PATHS ~/Library/Torch-Chemistry)
    list(APPEND hop_INCLUDE_DIRS ${tchem_INCLUDE_DIRS})
    list(APPEND hop_LIBRARIES ${tchem_LIBRARIES})
    set(hop_CXX_FLAGS "${tchem_CXX_FLAGS}")
endif()

# import location
find_library(hop_LIBRARY hop PATHS "${hopROOT}/lib")
set_target_properties(hop PROPERTIES
    IMPORTED_LOCATION "${hop_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${hop_INCLUDE_DIRS}"
    CXX_STANDARD 14
)