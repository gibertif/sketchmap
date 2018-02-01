cmake_minimum_required(VERSION 3.0)
project(sketchmap)

add_subdirectory(pybind11)
pybind11_add_module(sketchmap tools/main.cpp)

