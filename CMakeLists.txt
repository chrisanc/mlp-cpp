cmake_minimum_required(VERSION 3.10)
project(mpl-cpp)

add_executable(mlp
    main.cpp
    include/nn/nn.cpp
    include/math/math.cpp
    include/files/file.cpp
)

target_include_directories(mlp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)