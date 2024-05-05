cmake_minimum_required(VERSION 3.16)
project(lid_driven_gpu LANGUAGES CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -O3 -fopenmp)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_BUILD_TYPE Debug)


add_subdirectory($ENV{HOME}/medusa medusa)
include_directories($ENV{HOME}/medusa/include/)

include_directories($ENV{HOME}/libcudss-linux-x86_64-0.2.0.12_cuda12-archive/include)
link_directories($ENV{HOME}/libcudss-linux-x86_64-0.2.0.12_cuda12-archive/lib)


include_directories($ENV{HOME}/libcusparse_lt-linux-x86_64-0.6.0.6-archive/include)
link_directories($ENV{HOME}/libcusparse_lt-linux-x86_64-0.6.0.6-archive/lib)
# include_directories($ENV{HOME}/cusplibrary)

# required due to weird hdf5 installation
if (IS_DIRECTORY "/usr/include/hdf5/serial/")
    include_directories("/usr/include/hdf5/serial/")
elseif (IS_DIRECTORY "/usr/local/opt/hdf5/include/")
    include_directories("/usr/local/opt/hdf5/include/")
    link_directories("/usr/local/opt/hdf5/lib/")
endif ()

link_directories(/usr/lib/x86_64-linux-gnu/hdf5/serial/)

include_directories(src)

function(register name)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} medusa hdf5 gomp cusparse cudart cusolver cudss cusparseLt cublas)
    target_compile_options(${name} PRIVATE -fopenmp -std=c++17 -O3)
    target_include_directories(${name} PRIVATE src/solvers)
endfunction(register)
function(registerCuda name)
    add_executable(${name} src/${name}.cu)
    target_link_libraries(${name} medusa hdf5 gomp cusparse cudart cusolver cudss cublas)
    target_compile_options(${name} PRIVATE -fopenmp -std=c++17 -O3)
    target_include_directories(${name} PRIVATE src/solvers)
endfunction(registerCuda)
message("${CMAKE_CUDA_STANDARD}")
register(lidDriven)
register(lidDrivenACM)
register(lidDrivenMatrixACM)
registerCuda(lidDrivenMatrixACMGPU)
# registerCuda(lidDrivenIterativeGPU)