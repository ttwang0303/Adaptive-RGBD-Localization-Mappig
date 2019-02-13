cmake_minimum_required(VERSION 3.0)
project(adaptive-rgbd-localization)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -O3 -pthread" )

#------------------------------------------
#                    G2O
#------------------------------------------
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package( G2O REQUIRED )
include_directories(${G2O_INCLUDE_DIR})
find_package( Cholmod )
include_directories(${CHOLMOD_INCLUDE_DIR})
find_package(CSparse)
include_directories(${CSPARSE_INCLUDE_DIR})
list(APPEND G2O_LIBS
  g2o_cli g2o_ext_freeglut_minimal g2o_simulator
  g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
  g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
  g2o_types_sba g2o_types_slam3d g2o_csparse_extension
  g2o_opengl_helper g2o_solver_dense g2o_stuff
  g2o_types_sclam2d g2o_parser g2o_solver_pcg
  g2o_types_data g2o_types_sim3
)

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
find_package(PCL REQUIRED COMPONENTS common io registration filters)
add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARY_DIRS})

#------------------------------------------
#                    Pangolin
#------------------------------------------
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

#------------------------------------------
#                    DBoW2
#------------------------------------------
find_package(DBoW2 REQUIRED)
include_directories(${DBoW2_INCLUDE_DIRS})
find_package(DLib REQUIRED)
include_directories(${DLib_INCLUDE_DIRS})

#------------------------------------------
#                    Gnuplot-iostream
#------------------------------------------
#include_directories(/home/antonio/Public/Libraries/gnuplot-iostream/)

include_directories(${PROJECT_SOURCE_DIR})

add_library(${PROJECT_NAME} SHARED
Utils/utils.cpp
Utils/converter.cpp

Odometry/ransac.cpp
Odometry/generalizedicp.cpp
Odometry/adaptivergbdlocalization.cpp
Odometry/ransacpcl.cpp
Odometry/kabsch.cpp

Core/frame.cpp
Core/dbscan.cpp
Core/landmark.cpp
Core/map.cpp

Drawer/viewer.cpp
Drawer/pointclouddrawer.cpp

Features/matcher.cpp
Features/extractor.cpp
Features/detectoradjuster.cpp
Features/statefulfeaturedetector.cpp
Features/videodynamicadaptedfeaturedetector.cpp
Features/videogridadaptedfeaturedetector.cpp
)

#add_library(icp SHARED
#    Odometry/icp/icp.cpp
#    Odometry/icp/icpPointToPlane.cpp
#    Odometry/icp/icpPointToPoint.cpp
#    Odometry/icp/kdtree.cpp
#    Odometry/icp/matrix.cpp
#    )

#target_link_libraries(icp
#     ${BOOST_LIBRARIES}
#    )

target_link_libraries(${PROJECT_NAME}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${BOOST_LIBRARIES}
    ${G2O_LIBS}
    ${CHOLMOD_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${DLib_LIBS}
    ${DBoW2_LIBS})

add_executable(main main.cpp)
target_link_libraries(main ${PROJECT_NAME})
