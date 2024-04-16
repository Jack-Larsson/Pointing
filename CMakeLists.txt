cmake_minimum_required(VERSION 2.8)
project(KinectTest)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(k4a REQUIRED)

# Include Python headers
include_directories(KinectTest PRIVATE ${Python3_INCLUDE_DIRS})

# Add executable for C program
add_executable(KinectTest KinectTest.c)

# Link Python library
target_link_libraries(KinectTest PRIVATE Python3::Python)

# Link OpenCV libraries
target_link_libraries(KinectTest PRIVATE k4a::k4a ${OpenCV_LIBS})



CMake Error at CMakeLists.txt:9 (find_package):
  By not providing "Findk4a.cmake" in CMAKE_MODULE_PATH this project has
  asked CMake to find a package configuration file provided by "k4a", but
  CMake did not find one.

  Could not find a package configuration file provided by "k4a" with any of
  the following names:

    k4aConfig.cmake
    k4a-config.cmake

  Add the installation prefix of "k4a" to CMAKE_PREFIX_PATH or set "k4a_DIR"
  to a directory containing one of the above files.  If "k4a" provides a
  separate development package or SDK, be sure it has been installed.

