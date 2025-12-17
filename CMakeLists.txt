cmake_minimum_required(VERSION 3.16)
project(ml_dz_mf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(app
    src/main.cpp
    src/data_loader.cpp
    src/utils.cpp
    src/metrics.cpp
)
