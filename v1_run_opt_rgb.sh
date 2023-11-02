#!/bin/bash
GPU_NO=0;
is_bfm="False"

# # constants
basic_path=$(pwd)/3DMM/files/;
resources_path=$(pwd)/resources/;

uv_base="$basic_path/AI-NEXT-Albedo-Global.mat"
uv_regional_pyramid_base="$basic_path/AI-NEXT-AlbedoNormal-RPB/"

if [ $is_bfm == "False" ];then
    shape_exp_bases="$basic_path/AI-NEXT-Shape-NoAug.mat"
else
    shape_exp_bases="$resources_path/BFM2009_Model.mat"
fi

vggpath="$resources_path/vgg-face.mat"
pb_path=$resources_path/PB/

# # data directories
ROOT_DIR=$(pwd)/test_data/RGB/test1/single_img/;
img_dir=$ROOT_DIR

########################################################
echo "prepare datas";
cd ./data_prepare

prepare_dir="$ROOT_DIR/prepare_rgb"

python -u run_data_preparation.py \
        --GPU_NO=${GPU_NO}  \
        --mode='test_RGB' \
        --pb_path=${pb_path} \
        --img_dir=${img_dir} \
        --out_dir=${prepare_dir}

if [ "$?" -ne 0 ]; then echo "data prepare failed"; exit 1; fi