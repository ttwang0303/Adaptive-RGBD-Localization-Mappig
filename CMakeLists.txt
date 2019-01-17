cmake_minimum_required(VERSION 3.0)
project(adaptive-rgbd-localization)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -O3 -pthread" )

#------------------------------------------
#                    Boost
#------------------------------------------
FIND_PACKAGE( Boost REQUIRED )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

#------------------------------------------
#                    OpenCV
#------------------------------------------
find_package( OpenCV REQUIRED )
include_directories(${OPENCV_INCLUDE_DIRS})

#------------------------------------------
#                    Eigen
#------------------------------------------
find_package(Eigen3 3.0 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

#------------------------------------------
#                    PCL
#------------------------------------------
find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARY_DIRS})


include_directories(${PROJECT_SOURCE_DIR})

add_library(${PROJECT_NAME} SHARED
utils.cpp
converter.cpp
ransac.cpp
generalizedicp.cpp
adaptivergbdlocalization.cpp
frame.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${BOOST_LIBRARIES})

add_executable(main main.cpp)
target_link_libraries(main ${PROJECT_NAME})
