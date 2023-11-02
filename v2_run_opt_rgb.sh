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

cd ..

########################################################
echo "start RGB opt";

if [ $is_bfm == "False" ];then
    shape_out_dir=${ROOT_DIR}/our_opt_RGB
else
    shape_out_dir=${ROOT_DIR}/bfm_opt_RGB
fi


cd ./optimization/rgb

train_step=150
log_step=20
learning_rate=0.05
lr_decay_step=20
lr_decay_rate=0.9

photo_weight=100.0
gray_photo_weight=80.0
reg_shape_weight=0.5
reg_tex_weight=2.0
id_weight=1.0
real_86pt_lmk3d_weight=5.0
real_68pt_lmk2d_weight=5.0
lmk_struct_weight=0

num_of_img=1
project_type="Pers"

python run_RGB_opt.py \
--GPU_NO=${GPU_NO} \
--is_bfm=${is_bfm} \
--basis3dmm_path=${shape_exp_bases} \
--uv_path=${uv_base} \
--vggpath=${vggpath} \
--base_dir=${prepare_dir} \
--log_step=${log_step} \
--train_step=${train_step} \
--learning_rate=${learning_rate} \
--lr_decay_step=${lr_decay_step} \
--lr_decay_rate=${lr_decay_rate} \
--photo_weight=${photo_weight} \
--gray_photo_weight=${gray_photo_weight} \
--id_weight=${id_weight} \
--reg_shape_weight=${reg_shape_weight} \
--reg_tex_weight=${reg_tex_weight} \
--real_86pt_lmk3d_weight=${real_86pt_lmk3d_weight} \
--real_68pt_lmk2d_weight=${real_68pt_lmk2d_weight} \
--lmk_struct_weight=${lmk_struct_weight} \
--num_of_img=${num_of_img} \
--out_dir=${shape_out_dir} \
--summary_dir=${shape_out_dir}/summary \
--project_type=${project_type}

if [ "$?" -ne 0 ]; then echo "RGB opt failed"; exit 1; fi