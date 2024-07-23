cmake_minimum_required(VERSION 3.16)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# add_compile_options(-Wall -O3 -fopenmp)

function(register name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} medusa hdf5 gomp)
    target_compile_options(${name} PRIVATE -fopenmp -std=c++17 -O3)
endfunction(register)

register(lidDrivenACM)
register(lidDrivenMatrixACM)

# add_subdirectory(./cuda)
# add_subdirectory(./sycl)