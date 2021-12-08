cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpgpu LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

add_custom_target(etc-link
  [ ! -d etc ] && ln -s ${PROJECT_SOURCE_DIR}/etc ${PROJECT_BINARY_DIR}/etc ||
  exit 0
  )

set(SRC_GPU
  src/main.cu
  src/kernel.cu
  )

set(SRC_CPU
  src/main.cc
  src/kernel.cc
  )

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_source_files_properties(${SRC_CPU}
    PROPERTIES COMPILE_FLAGS "-g"
    )
endif ()

add_executable(gpgpu-gpu
  ${SRC_GPU}
  )

add_executable(gpgpu-cpu
  ${SRC_CPU}
  )

set(INCLUDE_COMMON
  "${PROJECT_SOURCE_DIR}/src/include"
  "${PROJECT_SOURCE_DIR}/extern/image"
  )

target_include_directories(gpgpu-gpu PUBLIC
  ${INCLUDE_COMMON}
  )

target_include_directories(gpgpu-cpu PUBLIC
  ${INCLUDE_COMMON}
  )

# RUN DEBUG TARGET
add_custom_target(run-gpu
  COMMAND printf %37sOUTPUT%37s\\\\n | tr " " "="
  COMMAND ./bin/gpgpu-gpu etc/data/01.jpg etc/data/result-gpu.jpg
  COMMAND printf %80s\\\\n | tr " " "="
  DEPENDS etc-link gpgpu-gpu
  )

add_custom_target(run-cpu
  COMMAND printf %37sOUTPUT%37s\\\\n | tr " " "="
  COMMAND ./bin/gpgpu-cpu etc/data/01.jpg etc/data/result-cpu.jpg
  COMMAND printf %80s\\\\n | tr " " "="
  DEPENDS etc-link gpgpu-cpu
  )

add_custom_target(run-profile-cpu
  COMMAND printf %37sOUTPUT%37s\\\\n | tr " " "="
  COMMAND valgrind --tool=callgrind --collect-systime=yes --separate-threads=yes --instr-atstart=no ./bin/gpgpu-cpu etc/data/00.jpg etc/data/result-cpu.jpg
  COMMAND printf %80s\\\\n | tr " " "="
  DEPENDS etc-link gpgpu-cpu
  )

set_property(TARGET gpgpu-gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
