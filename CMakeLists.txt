cmake_minimum_required(VERSION 3.18)

project(cvexec LANGUAGES CUDA)

# 设置 C++17 标准（NVCC 也支持）
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 强制构建类型为 Debug
set(CMAKE_BUILD_TYPE Debug)

# 设置 NVCC 编译选项：C++17、无优化、调试信息
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -g -G --std=c++17 -Xcompiler \"-O0 -g\"")

# 查找 OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

# 查找 FFmpeg（使用 pkg-config）
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)
pkg_check_modules(SWSCALE REQUIRED IMPORTED_TARGET libswscale)

set(SOURCES
    test_h_cpp/main.cpp
    test_h_cpp/cv_cess.cpp
    test_h_cpp/decode.cpp
)

# 创建可执行文件
add_executable(cvexec ${SOURCES})

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)

# 设置 CUDA 架构（根据你的 GPU 修改，例如 75 = Turing, 86 = Ampere）
set_property(TARGET cvexec PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)

# 链接 OpenCV 和 FFmpeg
target_link_libraries(cvexec
    ${OpenCV_LIBS}
    PkgConfig::AVCODEC
    PkgConfig::AVFORMAT
    PkgConfig::AVUTIL
    PkgConfig::SWSCALE
)

# 添加头文件目录（当前 CMakeLists.txt 所在目录下的 test_h_cpp）
target_include_directories(cvexec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_h_cpp)

# 可选：打印调试信息
message(STATUS " Building 'cvexec' with NVCC only (no explicit GCC)")
message(STATUS " CUDA Flags: ${CMAKE_CUDA_FLAGS_DEBUG}")
message(STATUS " OpenCV Version: ${OpenCV_VERSION}")
message(STATUS " Source Files: ${SOURCES}")
message(STATUS " Include Dir: ${CMAKE_CURRENT_SOURCE_DIR}/test_h_cpp")