cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpgpu LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_source_files_properties(${SRC}
    PROPERTIES COMPILE_FLAGS "-Wall -Wextra -pedantic -Werror -g -Wno-dev"
    )
endif ()

add_custom_target(etc-link
  [ ! -d etc ] && ln -s ${PROJECT_SOURCE_DIR}/etc ${PROJECT_BINARY_DIR}/etc ||
  exit 0
  )

set(SRC
  src/main.cu
  src/kernel.cu
  )

add_executable(gpgpu
  ${SRC}
  )

target_include_directories(gpgpu PUBLIC
  "${PROJECT_SOURCE_DIR}/src/include"
  "${PROJECT_SOURCE_DIR}/extern/image"
  )

# RUN DEBUG TARGET
add_custom_target(run
  COMMAND printf %37sOUTPUT%37s\\\\n | tr " " "="
  COMMAND ./bin/gpgpu etc/data/00.jpg etc/data/result.png
  COMMAND printf %80s\\\\n | tr " " "="
  DEPENDS etc-link gpgpu
  )


set_property(TARGET gpgpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
