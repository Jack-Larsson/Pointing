cmake_minimum_required(VERSION 2.8)
project(KinectTest)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(k4a REQUIRED)

# Include Python headers
include_directories(KinectTest PRIVATE ${Python3_INCLUDE_DIRS})

# Include Azure Kinect SDK headers
include_directories(AzureKinectTest PRIVATE ${k4a_INCLUDE_DIRS})

# Add executable for C program
add_executable(KinectTest KinectTest.c)

# Link Python library
target_link_libraries(KinectTest PRIVATE Python3::Python)

# Link OpenCV libraries
target_link_libraries(KinectTest PRIVATE k4a::k4a ${OpenCV_LIBS})

# Link Azure Kinect SDK libraries
target_link_libraries(AzureKinectTest PRIVATE ${k4a_LIBRARIES})