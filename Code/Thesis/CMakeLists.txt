# Requires Cmake version
cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

# Defines some variables
set( PROJ_NAME      "thesis" )
SET( PROJ_PATH      ${CMAKE_SOURCE_DIR} )
SET( PROJ_OUT_PATH  "build" )
SET( PROJ_LIBRARIES "" )
SET( PROJ_INCLUDES  "include" )
SET( PROJ_SRC       "src" )

# Set Project name
project(${PROJ_NAME} CXX)

# Find sources and headers
FILE( GLOB_RECURSE SOURCES ${PROJ_SRC}/*.cpp )
FILE( GLOB_RECURSE HEADERS ${PROJ_INCLUDES}/${PROJ_NAME}/*.h )

# Add include directory to the include path
INCLUDE_DIRECTORIES( ${PROJ_INCLUDES} )

# Add headers as custom targets
add_custom_target( headers SOURCES ${HEADERS} )

# Executables
add_executable(${PROJ_NAME} ${SOURCES})

# Compiler features
set_property(TARGET ${PROJ_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJ_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
