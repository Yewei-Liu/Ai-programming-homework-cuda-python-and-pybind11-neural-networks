# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# compile CUDA with /usr/bin/nvcc
CUDA_DEFINES = -Dmytensor_EXPORTS

CUDA_INCLUDES = -I/home/liuyewei/miniconda3/envs/backup/include -isystem=/home/liuyewei/miniconda3/envs/backup/include/python3.10 -isystem="/home/liuyewei/ai programming homework/final/task3/third_party/pybind11/include"

CUDA_FLAGS =  --generate-code=arch=compute_60,code=[compute_60,sm_60] --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75] --generate-code=arch=compute_80,code=[compute_80,sm_80] -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden

