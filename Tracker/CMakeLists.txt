cmake_minimum_required(VERSION 3.10)
project(run_bytetrack_demo)

# Use C++ 17 standard
set(CMAKE_CXX_STANDARD 17)

# Set TensorRT and CUDA library paths
find_package(CUDA REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
link_directories(/usr/local/cuda/lib64)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_library(NVONNXPARSER_LIBRARY nvonnxparser PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

# Set OpenCV library paths
find_package(OpenCV REQUIRED)
# Set Eigen3 library paths
find_package(Eigen3 REQUIRED)


# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS} 
    ${TENSORRT_INCLUDE_DIR} 
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/../Detector/opti_trt/include
    ${PROJECT_SOURCE_DIR}/../Detector/utils
    ${PROJECT_SOURCE_DIR}/bytetrack/include
    ${PROJECT_SOURCE_DIR}/demos
    )

# Project source files
file(GLOB OPTI_SOURCES ${PROJECT_SOURCE_DIR}/../Detector/opti_trt/src/*.cpp)
file(GLOB BYTETRACK_SOURCES ${PROJECT_SOURCE_DIR}/bytetrack/src/*.cpp)

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${OPTI_SOURCES}
    ${BYTETRACK_SOURCES}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${TENSORRT_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    ${CUDA_LIBRARIES}
    cudart
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# C++ compiler flags (optional)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-Wall")
