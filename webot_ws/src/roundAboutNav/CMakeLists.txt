cmake_minimum_required(VERSION 3.0.2)
project(roundAboutNav)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
)

#find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS date_time iostreams)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    geometry_msgs
    visualization_msgs
  # DEPENDS
  #   PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  # ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

link_directories(
  # ${PCL_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(round_about_node src/RoundAboutNav.cpp)
add_executable(arrow_cluster_node src/arrow_cluster_center.cpp)


target_link_libraries(round_about_node
  ${catkin_LIBRARIES}
  # ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

target_link_libraries(arrow_cluster_node
  ${catkin_LIBRARIES}
  # ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)
