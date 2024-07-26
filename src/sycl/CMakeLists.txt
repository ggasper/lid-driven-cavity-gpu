cmake_minimum_required(VERSION 3.20.5)


# set(CMAKE_CUDA_ARCHITECTURES 86)
#list(APPEND CMAKE_PREFIX_PATH $ENV{HOME}/intel/oneapi/2024.2/lib)
# list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
set(CMAKE_CXX_COMPILER icpx)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
# add_compile_options(-Wall -O3 -fopenmp)
# link_directories($ENV{CONDA_PREFIX}/lib/gcc/x86_64-conda-linux-gnu/12.4.0)
function(register name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} medusa hdf5 sycl pthread MKL::MKL)
    target_compile_options(${name} PRIVATE -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
    # target_include_directories(${name} PUBLIC $ENV{HOME}/intel/oneapi/2024.2/include $ENV{HOME}/intel/oneapi/2024.2/include/sycl --gcc-toolchain=$ENV{CONDA_PREFIX})
    # target_link_directories(${name} PRIVATE $ENV{HOME}/intel/oneapi/2024.2/lib)
endfunction(register)

register(lidDrivenACMSYCL)