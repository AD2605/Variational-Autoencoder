cmake_minimum_required(VERSION 3.16)
project(VAE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/home/atharva/torch/torch")
find_package(Torch REQUIRED)
add_executable(VAE main.cpp Autoencoder.h Autoencoder.cpp)
target_link_libraries(VAE "${TORCH_LIBRARIES}")