cmake_minimum_required(VERSION 3.10)
project(opencv_tracker)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.4.2 REQUIRED)
add_library(opencv INTERFACE)
target_include_directories(opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv INTERFACE ${OpenCV_LIBS})
message(STATUS "OpenCV library status:")
message(STATUS "    OPENCV_ROOT_DIR: ${OPENCV_ROOT_DIR}")
message(STATUS "    OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "    OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "    OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")


add_executable(circle_tracker circle_tracker.cpp)
target_link_libraries(circle_tracker
        opencv
        )

add_executable(mouse_tracker mouse_tracker.cpp)
target_link_libraries(mouse_tracker
        opencv
        )
