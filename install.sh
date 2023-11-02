#!/bin/bash
echo "compiling rasterizer"
TF_INC=/root/anaconda3/envs/v1/lib/python3.11/site-packages/tensorflow/include
TF_LIB=/root/anaconda3/envs/v1/lib/python3.11/site-packages/tensorflow
tf_mesh_renderer_path=$(pwd)/third_party/kernels/
g++ -std=c++17 \
    -shared $tf_mesh_renderer_path/rasterize_triangles_grad.cc $tf_mesh_renderer_path/rasterize_triangles_op.cc $tf_mesh_renderer_path/rasterize_triangles_impl.cc \
    -o $tf_mesh_renderer_path/rasterize_triangles_kernel.so -fPIC \
    -I$TF_INC -L$TF_LIB -l:libtensorflow_framework.so.2 -l:libtensorflow_cc.so.2 -O2

if [ "$?" -ne 0 ]; then echo "compile rasterizer failed"; exit 1; fi
