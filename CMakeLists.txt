cmake_minimum_required(VERSION 3.0)
project(KinectTest)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add executable for C program
add_executable(KinectTest KinectTest.c)

# Link OpenCV libraries
target_link_libraries(KinectTest PRIVATE ${OpenCV_LIBS})

# Include Python headers
target_include_directories(KinectTest PRIVATE ${Python3_INCLUDE_DIRS})

# Link Python library
target_link_libraries(KinectTest PRIVATE Python3::Python)
