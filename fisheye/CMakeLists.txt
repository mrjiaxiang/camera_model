cmake_minimum_required(VERSION 2.8.3)
project(pinhole_model)

set(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_compile_options(-std=c++14)
add_definitions(-std=c++14)

set(ALL_TARGET_LIBRARIES "")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pinhole_node src/main.cpp src/function.cpp)
target_link_libraries(pinhole_node ${OpenCV_LIBRARIES} glog)