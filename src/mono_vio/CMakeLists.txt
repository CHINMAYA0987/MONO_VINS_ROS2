cmake_minimum_required(VERSION 3.8)
project(mono_vio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

add_library(mono_vio_lib
  src/imu_integrator.cpp
  src/feature_tracker.cpp
  src/initializer.cpp
  src/sliding_window.cpp
  src/marginalization.cpp
  src/optimizer.cpp
  src/triangulator.cpp
  src/loop_detector.cpp
)

target_include_directories(mono_vio_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(mono_vio_lib
  rclcpp sensor_msgs geometry_msgs nav_msgs cv_bridge tf2_ros tf2_eigen)

target_link_libraries(mono_vio_lib
  Eigen3::Eigen
  ${CERES_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(mono_vio_node src/mono_vio_node.cpp)
target_link_libraries(mono_vio_node mono_vio_lib)
ament_target_dependencies(mono_vio_node
  rclcpp sensor_msgs geometry_msgs nav_msgs cv_bridge tf2_ros tf2_eigen)

install(TARGETS mono_vio_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})

ament_package()