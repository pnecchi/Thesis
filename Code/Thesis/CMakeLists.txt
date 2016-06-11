cmake_minimum_required(VERSION 2.8)

project(thesis)

# ----------------------- INCLUDES ----------------------------
include_directories(include)


# ----------------------- LIBRARIES ---------------------------
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIR})

# ----------------------- GCC FLAGS ----------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(BUILD_TYPE_MSG "Debug")

else()
    set(BUILD_TYPE_MSG "Release")
endif()

# ------------------------ MESSAGES ----------------------------

message(STATUS "Build type       : " ${BUILD_TYPE_MSG})

# ------------------------ BUILD -------------------------------

file(GLOB_RECURSE PRJ_SOURCE src/*.cpp)

file(GLOB_RECURSE PRJ_INCLUDE include/*.h)

add_library(${PROJECT_NAME} ${PRJ_SOURCE} ${PRJ_INCLUDE})
target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

# examples folder contains the executable files
add_subdirectory(examples)
