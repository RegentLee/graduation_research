cmake_minimum_required(VERSION 3.14)
project(graduation_research)

set(CMAKE_CXX_STANDARD 14)

add_executable(graduation_research main.cpp csv.cpp csv.h SDNN.cpp SDNN.h Trainer.cpp Trainer.h Tester.cpp Tester.h SDNNOpenMP.cpp SDNNOpenMP.h SDNNBiOpenMP.cpp SDNNBiOpenMP.h func.cpp func.h SDNNFABiOpenMP.cpp SDNNFABiOpenMP.h SDNNWithUpdateOpenMP.cpp SDNNWithUpdateOpenMP.h)