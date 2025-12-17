cmake_minimum_required(VERSION 3.16)
project(HW_var22 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# По умолчанию Release, если не задано
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Собираем все исходники из src/
file(GLOB SRC_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(app ${SRC_FILES})

target_include_directories(app PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Предупреждения (аккуратно, без -Werror)
if (MSVC)
  target_compile_options(app PRIVATE /W4)
else()
  target_compile_options(app PRIVATE -Wall -Wextra -Wpedantic)
endif()
