cmake_minimum_required(VERSION 2.8.3)
project(vision-based-avoidance)
#find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
 #       sensor_msgs
 #       std_msgs
 #       geometry_msgs
 #       cv_bridge
 #       image_transport
        )

## declare and generate any messages or services

## Declare catkin package
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES
    CATKIN_DEPENDS roscpp rospy std_msgs
)


## Header files to include
#add_library(find_ball_lib STATIC src/find_ball.h src/find_ball.cpp)
#install(TARGETS find_ball_lib DESTINATION include)

## Build nodes
include_directories(include)
include_directories(include ${catkin_INCLUDE_DIRS})


##add_executable(compress_test src/compress_test.cpp)
##target_link_libraries(compress_test ${OpenCV_LIBS} ${catkin_LIBRARIES})
