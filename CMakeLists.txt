cmake_minimum_required(VERSION 3.0.2)
project(gaussian_lic)

SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -Wno-sign-compare -Wno-unused -Wno-comment -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")

if(NOT OpenCV_DIR)
  set(OpenCV_DIR "$ENV{HOME}/Software/opencv/opencv-4.7.0/build")
endif()

if(NOT Torch_DIR)
  set(Torch_DIR "$ENV{HOME}/Software/libtorch/share/cmake/Torch")
endif()

if(NOT TENSORRT_ROOT)
  set(TENSORRT_ROOT "$ENV{HOME}/Software/TensorRT-8.6.1.6")
endif()

if(NOT CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.7")
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  std_msgs
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf
  eigen_conversions
)

find_package(OpenCV 4 REQUIRED)
find_package(Torch REQUIRED)

set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")
if(NOT EXISTS "${TENSORRT_LIB_DIR}")
  set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
endif()

set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
find_package(CUDA REQUIRED)
find_library(FFI_LIB ffi)

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gaussian_lic
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  src
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

add_executable(test_language_targets
              ${PROJECT_SOURCE_DIR}/src/test_language_targets.cpp
              ${PROJECT_SOURCE_DIR}/src/language_targets.cpp)
target_link_libraries(test_language_targets
                      ${OpenCV_LIBS}
                      ${TORCH_LIBRARIES}
                      ${YAML_CPP_LIBRARIES})

add_executable(gs_mapping 
              ${PROJECT_SOURCE_DIR}/src/mapping.cpp 
              ${PROJECT_SOURCE_DIR}/src/gaussian.cpp
              ${PROJECT_SOURCE_DIR}/src/tinyply.cpp
              ${PROJECT_SOURCE_DIR}/src/depth_completer.cpp 
              ${PROJECT_SOURCE_DIR}/src/simple-knn/spatial.cu
              ${PROJECT_SOURCE_DIR}/src/simple-knn/simple_knn.cu
              ${PROJECT_SOURCE_DIR}/src/fused-ssim/ssim.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/renderer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterizer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterize_points.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/adam.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/forward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/backward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/rasterizer_impl.cu)
set_target_properties(gs_mapping PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED OFF)
target_link_libraries(gs_mapping 
                      ${catkin_LIBRARIES} 
                      ${OpenCV_LIBS} 
                      ${TORCH_LIBRARIES} 
                      ${YAML_CPP_LIBRARIES} 
                      ${Python3_LIBRARIES}
                      ${FFI_LIB}
                      ${CUDA_LIB_DIR}/libcudart.so
                      ${TENSORRT_LIB_DIR}/libnvinfer.so
                      ${TENSORRT_LIB_DIR}/libnvonnxparser.so
                      ${TENSORRT_LIB_DIR}/libnvparsers.so
                      ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so)

add_executable(gs_colmap
              ${PROJECT_SOURCE_DIR}/src/colmap_loader.cpp
              ${PROJECT_SOURCE_DIR}/src/gaussian.cpp
              ${PROJECT_SOURCE_DIR}/src/tinyply.cpp
              ${PROJECT_SOURCE_DIR}/src/depth_completer.cpp
              ${PROJECT_SOURCE_DIR}/src/simple-knn/spatial.cu
              ${PROJECT_SOURCE_DIR}/src/simple-knn/simple_knn.cu
              ${PROJECT_SOURCE_DIR}/src/fused-ssim/ssim.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/renderer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterizer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterize_points.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/adam.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/forward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/backward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/rasterizer_impl.cu)
set_target_properties(gs_colmap PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED OFF)
target_link_libraries(gs_colmap
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${TORCH_LIBRARIES}
                      ${YAML_CPP_LIBRARIES}
                      ${Python3_LIBRARIES}
                      ${FFI_LIB}
                      ${CUDA_LIB_DIR}/libcudart.so
                      ${TENSORRT_LIB_DIR}/libnvinfer.so
                      ${TENSORRT_LIB_DIR}/libnvonnxparser.so
                      ${TENSORRT_LIB_DIR}/libnvparsers.so
                      ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so)

add_executable(gs_fastlivo_incremental
              ${PROJECT_SOURCE_DIR}/src/fastlivo_incremental.cpp
              ${PROJECT_SOURCE_DIR}/src/language_targets.cpp
              ${PROJECT_SOURCE_DIR}/src/gaussian.cpp
              ${PROJECT_SOURCE_DIR}/src/tinyply.cpp
              ${PROJECT_SOURCE_DIR}/src/depth_completer.cpp
              ${PROJECT_SOURCE_DIR}/src/simple-knn/spatial.cu
              ${PROJECT_SOURCE_DIR}/src/simple-knn/simple_knn.cu
              ${PROJECT_SOURCE_DIR}/src/fused-ssim/ssim.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/renderer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterizer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterize_points.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/adam.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/forward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/backward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/rasterizer_impl.cu)
set_target_properties(gs_fastlivo_incremental PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED OFF)
target_link_libraries(gs_fastlivo_incremental
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${TORCH_LIBRARIES}
                      ${YAML_CPP_LIBRARIES}
                      ${Python3_LIBRARIES}
                      ${FFI_LIB}
                      ${CUDA_LIB_DIR}/libcudart.so
                      ${TENSORRT_LIB_DIR}/libnvinfer.so
                      ${TENSORRT_LIB_DIR}/libnvonnxparser.so
                      ${TENSORRT_LIB_DIR}/libnvparsers.so
                      ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so)
