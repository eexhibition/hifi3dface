"""
This file is part of the repo: https://github.com/tencent-ailab/hifi3dface

If you find the code useful, please cite our paper: 

"High-Fidelity 3D Digital Human Head Creation from RGB-D Selfies."
ACM Transactions on Graphics 2021
Code: https://github.com/tencent-ailab/hifi3dface

Copyright (c) [2020-2021] [Tencent AI Lab]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
import numpy as np
import cv2
import sys
import os

sys.path.append("..")

from third_party.rasterize_triangles import rasterize_clip_space
from .basis import load_3dmm_basis, get_geometry, get_region_uv_texture, get_texture


class Shader(object):
    def __init__(self):
        pass

    @staticmethod
    def _lambertian_attentuation():
        """ constant weight from sfsnet matlab """
        return np.pi * np.array([1, 2.0 / 3, 1.0 / 4])

    @staticmethod
    def _harmonics(ver_norm, order=2):
        """compute the spherical harmonics function for 3D vertices.
        :param:
            ver_norm: [batch, N, 3], vertex normal

        :return:
            H: [batch, 9], 2-order harmonic basis
        """
        lam_attn = Shader._lambertian_attentuation()

        x, y, z = tf.split(ver_norm, 3, -1)
        x2 = tf.square(x)
        y2 = tf.square(y)
        z2 = tf.square(z)
        xy = x * y
        yz = y * z
        xz = x * z
        PI = np.pi

        l0 = np.sqrt(1.0 / (4 * PI)) * tf.ones_like(x)
        l1x = np.sqrt(3.0 / (4 * PI)) * x
        l1y = np.sqrt(3.0 / (4 * PI)) * y
        l1z = np.sqrt(3.0 / (4 * PI)) * z
        l2xy = np.sqrt(15.0 / (4 * PI)) * xy
        l2yz = np.sqrt(15.0 / (4 * PI)) * yz
        l2xz = np.sqrt(15.0 / (4 * PI)) * xz
        l2z2 = np.sqrt(5.0 / (16 * PI)) * (3 * z2 - 1)
        l2x2_y2 = np.sqrt(15.0 / (16 * PI)) * (x2 - y2)
        H = tf.concat(
            [l0, l1z, l1x, l1y, l2z2, l2xz, l2yz, l2x2_y2, l2xy],
            -1,
            name="hamonics_basis_order2",
        )
        if order == 3:
            b9 = 1.0 / 4.0 * np.sqrt(35.0 / (2 * PI)) * (3 * x2 - z2) * z
            b10 = 1.0 / 2.0 * np.sqrt(105.0 / PI) * x * yz
            b11 = 1.0 / 4 * np.sqrt(21.0 / (2 * PI)) * z * (4 * y2 - x2 - z2)
            b12 = 1.0 / 4 * np.sqrt(7.0 / PI) * y * (2 * y2 - 3 * x2 - 3 * z2)
            b13 = 1.0 / 4 * np.sqrt(21.0 / (2 * PI)) * x * (4 * y2 - x2 - z2)
            b14 = 1.0 / 4 * np.sqrt(105.0 / PI) * (x2 - z2) * y
            b15 = 1.0 / 4 * np.sqrt(35.0 / (2 * PI)) * (x2 - 3 * z2) * x
            H = tf.concat(
                [H, b9, b10, b11, b12, b13, b14, b15], -1, name="harmonics_basis_order3"
            )
        batch_size, img_height, img_width, _ = ver_norm.get_shape().as_list()
        H.set_shape([batch_size, img_height, img_width, 9])
        return H

    @staticmethod
    def sh_shader(normals, alphas, background_images, sh_coefficients, diffuse_colors):
        """
        render mesh into image space and return all intermediate results.
        :param:
            normals: [batch,300,300,3], vertex normals in image space
            alphas: [batch,H,W,1], alpha channels
            background_images: [batch,H,W,3], background images for rendering results
            sh_coefficient: [batch,27], 2-order SH coefficient
            diffuse_colors: [batch,H,W,3], vertex colors in image space

        sh_coefficient: [batch_size, 27] spherical harmonics coefficients.
        """
        batch_size, image_height, image_width = [s.value for s in normals.shape[:-1]]

        sh_coef_count = sh_coefficients.get_shape().as_list()[-1]

        if sh_coef_count == 27:
            init_para_illum = tf.constant([1] + [0] * 8, tf.float32, name="init_illum")
            init_para_illum = tf.reshape(
                init_para_illum, [1, 9], name="init_illum_reshape"
            )
            init_para_illum = tf.concat(
                [init_para_illum] * 3, axis=1, name="init_illum_concat"
            )
            sh_coefficients = sh_coefficients + init_para_illum  # batch x 27
            order = 2
        else:
            init_para_illum = tf.constant([1.0] * 2 + [0] * 14, tf.float32)
            init_para_illum = tf.reshape(init_para_illum, [1, 16])
            init_para_illum = tf.concat([init_para_illum] * 3, axis=1)
            sh_coefficients = sh_coefficients + init_para_illum
            sh_coefficients = tf.tile(
                tf.reshape(sh_coefficients, [-1, 1, 1, 3, 16]),
                [1, image_height, image_width, 1, 1],
            )
            order = 3

        batch_size = diffuse_colors.get_shape().as_list()[0]
        sh_kernels = tf.split(sh_coefficients, batch_size, axis=0)
        harmonic_output = Shader._harmonics(normals, order)
        harmonic_output_list = tf.split(harmonic_output, batch_size, axis=0)
        results = []
        for ho, shk in zip(harmonic_output_list, sh_kernels):
            shk = tf.reshape(tf.transpose(tf.reshape(shk, [3, 9])), [1, 1, 9, 3])
            res = tf.nn.conv2d(ho, shk, strides=[1, 1, 1, 1], padding="SAME")
            results.append(res)
        shading = tf.concat(results, axis=0)

        rgb_images = shading * diffuse_colors

        alpha_images = tf.reshape(
            alphas, [-1, image_height, image_width, 1], name="alpha_images"
        )
        valid_rgb_values = tf.concat(
            3 * [alpha_images > 0.5], axis=3, name="valid_rgb_values"
        )

        rgb_images = tf.where(
            valid_rgb_values, rgb_images, background_images, name="rgb_images"
        )

        return rgb_images, shading

    @staticmethod
    def remove_shading(images, image_normals, sh_coefficients):

        init_para_illum = tf.constant([1] + [0] * 8, tf.float32)
        init_para_illum = tf.reshape(init_para_illum, [1, 9])
        init_para_illum = tf.concat([init_para_illum] * 3, axis=1)
        sh_coefficients = sh_coefficients + init_para_illum  # careful

        _, image_height, image_width = [s.value for s in image_normals.shape[:-1]]
        sh_coefficients = tf.tile(
            tf.reshape(sh_coefficients, [-1, 1, 1, 3, 9]),
            [1, image_height, image_width, 1, 1],
        )
        harmonic_output = tf.expand_dims(Shader._harmonics(image_normals), -1)
        shading = tf.squeeze(tf.matmul(sh_coefficients, harmonic_output))
        diffuse_maps = images / (shading + 1e-18)
        return diffuse_maps


class Projector(object):
    def __init__(self):
        pass

    @staticmethod
    def get_ver_norm(ver_xyz, tri, scope_name="normal"):
        """
        Compute vertex normals.

        :param ver_xyz: [batch, N, 3], vertex geometry
        :param tri: [M, 3], mesh triangles definition

        :return: ver_normals: [batch, N, 3], vertex normals
        """

        with tf.name_scope(scope_name):
            """
            Extracting the vertices for each triangle
            1. 삼각형의 꼭짓점 가져오기
            이 부분에서는 메쉬를 구성하는 삼각형의 꼭짓점 좌표를 가져옵니다.
            tri 텐서는 각 삼각형을 구성하는 꼭짓점의 인덱스를 포함하고 있습니다.
            tf.gather를 사용하여 이 인덱스에 해당하는 실제 꼭짓점의 좌표를 가져옵니다.
            """
            v1_idx, v2_idx, v3_idx = tf.unstack(tri, 3, axis=-1)
            v1 = tf.gather(ver_xyz, v1_idx, axis=1, name="v1_tri")
            v2 = tf.gather(ver_xyz, v2_idx, axis=1, name="v2_tri")
            v3 = tf.gather(ver_xyz, v3_idx, axis=1, name="v3_tri")

            # Calculating the normals for each triangle
            """
            2. 삼각형의 표면 벡터 계산
            삼각형의 두 변을 사용하여 외적(cross product)을 계산하고, 이를 통해 삼각형의 표면 벡터(정규벡터)를 구합니다.
            EPS는 0으로 나누는 것을 방지하기 위한 작은 상수입니다.
            결과 벡터를 정규화하여 길이가 1이 되도록 합니다.
            """
            EPS = 1e-8
            tri_normals = tf.linalg.cross(v2 - v1, v3 - v1)
            tri_normals = tf.divide(
                tri_normals,
                (tf.norm(tri_normals, axis=-1, keepdims=True) + EPS),
                name="norm_tri",
            )

            # Accumulating vertex normals
            """
            3. 꼭짓점의 정규벡터 계산
            삼각형의 각 꼭짓점에 대해 해당 삼각형의 정규벡터를 할당합니다.
            이를 위해 텐서의 형태를 변형하여 각 꼭짓점에 대해 동일한 정규벡터가 반복되도록 합니다.
            """
            tri_normals = tf.tile(tf.expand_dims(tri_normals, 2), [1, 1, 3, 1])
            tri_normals = tf.reshape(tri_normals, [-1, 3])
            tri_votes = tf.cast(tf.greater(tri_normals[:, 2:], float(0.1)), tf.float32)
            tri_cnts = tf.ones_like(tri_votes)

            """
            4. 정규벡터의 투표
            각 꼭짓점에 대해 투표를 수행합니다.
            tri_inds 텐서는 각 꼭짓점의 인덱스와 해당 인덱스가 속한 배치의 인덱스를 포함합니다.
            """
            B = v1.get_shape().as_list()[0]  # batch size
            tri_shape = tf.shape(tri)
            tri_len = tri_shape[0]  # Or whatever dimension you are interested in

            batch_indices = tf.reshape(
                tf.tile(tf.expand_dims(tf.range(B), axis=1), [1, tri_len * 3]),
                [-1],
                name="batch_indices",
            )
            tri_inds = tf.stack(
                [
                    batch_indices,
                    tf.concat([tf.reshape(tri, [tri_len * 3])] * B, axis=0),
                ],
                axis=1,
            )

            """
            5. 투표 결과를 사용하여 꼭짓점의 정규벡터 계산
            tf.tensor_scatter_nd_add를 사용하여 각 꼭짓점에 대해 투표된 정규벡터를 더합니다.
            이후 정규벡터를 정규화하여 길이를 1로 만듭니다.
            
            ver_normals = tf.Variable(tf.zeros(ver_shape), trainable=False)
            ver_normals = tf.tensor_scatter_nd_add(ver_normals, tri_inds, tri_normals)
            ver_normals = ver_normals / (tf.norm(ver_normals, axis=2, keepdims=True) + EPS)
            """
            ver_xyz = tf.convert_to_tensor(ver_xyz)
            ver_shape = ver_xyz.shape.as_list()

            ver_normals = tf.Variable(
                initial_value=tf.zeros(ver_shape, dtype=tf.float32),
                trainable=False,
                name="ver_norm"
            )

            init_normals = tf.zeros(ver_shape, dtype=tf.float32)
            ver_normals.assign(init_normals)
            ver_normals.assign_add(tf.tensor_scatter_nd_add(ver_normals, tri_inds, tri_normals))
            ver_normals.assign(ver_normals / (tf.norm(ver_normals, axis=2, keepdims=True) + EPS))

            """
            6. 윤곽선 마스크 계산
            ver_votes는 꼭짓점이 얼마나 많이 "투표"를 받았는지를 나타냅니다.
            ver_cnts는 해당 꼭짓점에 투표된 횟수를 나타냅니다.
            마지막으로, 각 꼭짓점의 투표 결과를 횟수로 나누어 평균을 구합니다.
            윤곽선 마스크는 투표 결과가 0보다 크고 1보다 작은 꼭짓점을 찾아서 계산합니다.
            """
            ver_votes = tf.Variable(
                initial_value=tf.zeros(ver_shape[:-1] + [1], dtype=tf.float32),
                trainable=False,
                name="ver_vote"
            )
            ver_cnts = tf.Variable(
                initial_value=tf.zeros(ver_shape[:-1] + [1], dtype=tf.float32),
                trainable=False,
                name="ver_cnt"
            )

            init_votes = tf.zeros(ver_shape[:-1] + [1], dtype=tf.float32)
            ver_votes.assign(init_votes)
            ver_cnts.assign(init_votes)

            ver_votes.assign_add(tf.tensor_scatter_nd_add(ver_votes, tri_inds, tri_votes))
            ver_cnts.assign_add(tf.tensor_scatter_nd_add(ver_cnts, tri_inds, tri_cnts))
            ver_votes.assign(ver_votes / (ver_cnts + EPS))

            ver_votes1 = tf.less(ver_votes, float(1.0))
            ver_votes2 = tf.greater(ver_votes, float(0.0))
            ver_votes.assign(tf.cast(tf.logical_and(ver_votes1, ver_votes2), tf.float32))

            return ver_normals, ver_votes

    @staticmethod
    def generate_base_information(basis3dmm, para_shape, para_tex):
        vt_list = basis3dmm["vt_list"]
        tri = basis3dmm["tri"]
        tri_vt = basis3dmm["tri_vt"]
        tri = tri.astype(np.int32)
        tri_vt = tri_vt.astype(np.int32)

        ver_xyz = get_geometry(basis3dmm, para_shape)  # 1, 20481, 3

        uv_rgb, uv_mask = get_region_uv_texture(basis3dmm["uv"], para_tex, uv_size=512)
        uv_rgb = uv_rgb / 255.0

        return tri, tri_vt, vt_list, uv_rgb, uv_mask, ver_xyz

    @staticmethod
    def generate_base_information_BFM(basis3dmm, para_shape, para_tex):
        tri = basis3dmm["tri"]
        tri = tri.astype(np.int32)

        ver_xyz = get_geometry(basis3dmm, para_shape)  # 1, 20481, 3
        ver_rgb = get_texture(basis3dmm, para_tex)  # 1, 20481, 3
        ver_rgb = tf.clip_by_value(ver_rgb / 255.0, 0.0, 1.0)
        batch_size, _, _ = ver_xyz.get_shape().as_list()
        ver_mask = tf.concat(
            [np.reshape(basis3dmm["mask_face"], [1, -1, 1])] * batch_size,
            axis=0,
            name="ver_face_mask",
        )  # 1, 20481, 1
        ver_mask = tf.cast(ver_mask, tf.float32)
        return tri, ver_rgb, ver_mask, ver_xyz

    @staticmethod
    def generate_proj_information(
            ver_xyz,
            trans_Mat,
            K_img,
            imageH,
            imageW,
            tri,
            project_type="Pers",
            name="ver_norm_and_ver_depth",
    ):

        ver_w = tf.ones_like(ver_xyz[:, :, 0:1], name="ver_w")
        ver_xyzw = tf.concat([ver_xyz, ver_w], axis=2)  # 1, 20481, 4
        vertex_img = ver_xyzw @ trans_Mat  # 1 x 20481 x 4
        cam_xyz = vertex_img[:, :, 0:3]  # 1 x 20481 x 3

        K_img = tf.transpose(K_img, [0, 2, 1])  # 1 x 3 x 3
        proj_xyz_batch = cam_xyz @ K_img  # 1 x 20481 x 3
        proj_xyz_depth_batch = cam_xyz @ K_img  # 1 x 20481 x 3

        if project_type == "Orth":
            clip_x = ((proj_xyz_batch[:, :, 0] + imageW / 2) / imageW * 2 - 1)[:, :, None]
            clip_y = ((proj_xyz_batch[:, :, 1] + imageH / 2) / imageH * 2 - 1)[:, :, None]
        else:
            clip_x = ((proj_xyz_batch[:, :, 0] / proj_xyz_batch[:, :, 2]) / imageW * 2 - 1)[:, :, None]
            clip_y = ((proj_xyz_batch[:, :, 1] / proj_xyz_batch[:, :, 2]) / imageH * 2 - 1)[:, :, None]
        clip_z = tf.nn.l2_normalize(proj_xyz_batch[:, :, 2], axis=1, epsilon=1e-10)[:, :, None]

        clip_xyz = tf.concat([clip_x, clip_y, clip_z], axis=2)  # 1, 20481, 3
        clip_w = tf.ones_like(clip_xyz[:, :, 0:1], name="clip_w")
        clip_xyzw = tf.concat([clip_xyz, clip_w], axis=2)  # 1, 20481, 4

        if project_type == "Orth":
            proj_x = (proj_xyz_batch[:, :, 0] + imageW / 2)[:, :, None]
            proj_y = (proj_xyz_batch[:, :, 1] + imageH / 2)[:, :, None]
        else:
            proj_x = (proj_xyz_batch[:, :, 0] / proj_xyz_batch[:, :, 2])[:, :, None]
            proj_y = (proj_xyz_batch[:, :, 1] / proj_xyz_batch[:, :, 2])[:, :, None]
        proj_z = proj_xyz_batch[:, :, 2][:, :, None]
        proj_xy = tf.concat([proj_x, proj_y], axis=2)  # 1, 20481, 2

        depth_infor = proj_xyz_depth_batch[:, :, 2][:, :, None]

        # NOTE: The function get_ver_norm is assumed to be defined elsewhere
        ver_norm, ver_contour_mask = Projector.get_ver_norm(cam_xyz, tri)
        norm_depth_infro = tf.concat([ver_norm, depth_infor, ver_contour_mask], axis=2)  # 1, 20481, 4

        norm_depth_image, alphas = rasterize_clip_space(
            clip_xyzw, norm_depth_infro, tri, imageW, imageH, 0.0
        )
        norm_image = norm_depth_image[:, :, :, 0:3]
        depth_image = norm_depth_image[:, :, :, 3:4]
        ver_contour_mask_image = norm_depth_image[:, :, :, 4:5]

        return (
            norm_image,
            ver_norm,
            alphas,
            clip_xyzw,
            proj_xy,
            proj_z,
            depth_image,
            ver_contour_mask,
            ver_contour_mask_image,
        )

    @staticmethod
    def project_uv_render(
        ori_img,
        norm_image,
        clip_xyzw,
        tri,
        tri_vt,
        vt_list,
        imageH,
        imageW,
        uv_rgb,
        uv_mask,
        para_illum,
        var_scope_name,
    ):
        batch_size, _, _ = clip_xyzw.get_shape().as_list()
        # get uv coordinates
        V, U = tf.split(vt_list, 2, axis=1)
        uv_size = uv_rgb.get_shape().as_list()[1]
        U = (1.0 - U) * uv_size
        V = V * uv_size
        UV = tf.concat([U, V], axis=1)
        batch_UV = tf.tile(UV, [batch_size, 1])

        # get clip_xyzw for ver_uv (according to the correspondence between tri and tri_vt)
        # gather and scatter
        EPS = 1e-12
        batch_tri_indices = tf.reshape(
            tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, len(tri_vt) * 3]),
            [-1],
            name="batch_tri_indices",
        )
        tri_inds = tf.stack(
            [
                batch_tri_indices,
                tf.concat([tf.reshape(tri, [len(tri) * 3])] * batch_size, axis=0),
            ],
            axis=1,
        )
        tri_vt_inds = tf.stack(
            [
                batch_tri_indices,
                tf.concat([tf.reshape(tri_vt, [len(tri_vt) * 3])] * batch_size, axis=0),
            ],
            axis=1,
        )
        tri_clip_xyzw = tf.gather_nd(clip_xyzw, tri_inds, name="tri_clip_xyzw")
        ver_uv_clip_xyzw_sum = tf.get_variable(
            shape=[batch_size, len(vt_list), 4],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            name=var_scope_name + "ver_uv_clip_xyzw_sum",
            trainable=False,
        )
        ver_uv_clip_xyzw_cnt = tf.get_variable(
            shape=[batch_size, len(vt_list), 4],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            name=var_scope_name + "ver_uv_clip_xyzw_cnt",
            trainable=False,
        )
        init_ver_uv = tf.zeros(shape=[batch_size, len(vt_list), 4], dtype=tf.float32)
        assign_op1 = tf.assign(ver_uv_clip_xyzw_sum, init_ver_uv)
        assign_op2 = tf.assign(ver_uv_clip_xyzw_cnt, init_ver_uv)
        with tf.control_dependencies([assign_op1, assign_op2]):
            ver_uv_clip_xyzw_sum = tf.scatter_nd_add(
                ver_uv_clip_xyzw_sum, tri_vt_inds, tri_clip_xyzw
            )
            ver_uv_clip_xyzw_cnt = tf.scatter_nd_add(
                ver_uv_clip_xyzw_cnt, tri_vt_inds, tf.ones_like(tri_clip_xyzw)
            )
            ver_uv_clip_xyzw = tf.div(ver_uv_clip_xyzw_sum, ver_uv_clip_xyzw_cnt + EPS)

            uv_image, uv_alphas = rasterize_clip_space(
                ver_uv_clip_xyzw, batch_UV, tri_vt, imageW, imageH, -1.0
            )

            uv_image = tf.clip_by_value(
                tf.cast(uv_image, tf.int32), 0, 511
            )  # should be integer

            batch_vt_indices = tf.reshape(
                tf.tile(
                    tf.expand_dims(tf.range(batch_size), axis=1), [1, imageW * imageH]
                ),
                [-1, 1],
                name="batch_indices",
            )

            batch_vt_indices = tf.concat(
                [batch_vt_indices, tf.reshape(uv_image, [-1, 2])], axis=1
            )

            # careful
            diffuse_image = tf.reshape(
                tf.gather_nd(uv_rgb, batch_vt_indices), [batch_size, imageH, imageW, 3]
            )
            uv_alphas = (
                tf.reshape(
                    tf.gather_nd(uv_mask[:, :, :, 0], batch_vt_indices),
                    [batch_size, imageH, imageW, 1],
                )
                * uv_alphas
            )

        # Have shading
        para_light = para_illum
        background = ori_img
        rgb_images, shading_image = Shader.sh_shader(
            norm_image, uv_alphas, background, para_light, diffuse_image
        )
        ori_img_remove_shading = ori_img / shading_image

        diffuse_image = tf.clip_by_value(diffuse_image, 0, 1)
        rgb_images = tf.clip_by_value(rgb_images, 0, 1)
        uv_attrs_image = tf.clip_by_value(uv_alphas, 0, 1)
        ori_img_remove_shading = tf.clip_by_value(ori_img_remove_shading, 0, 1)

        render_image = rgb_images
        render_image = render_image * uv_attrs_image + ori_img * (1 - uv_attrs_image)

        return render_image, uv_attrs_image, ori_img_remove_shading

    @staticmethod
    def project_vertex_render(
        ori_img,
        norm_image,
        clip_xyzw,
        tri,
        imageH,
        imageW,
        ver_rgb,
        ver_mask,
        para_illum,
        var_scope_name,
    ):
        with tf.variable_scope(var_scope_name):
            batch_size, _, _ = clip_xyzw.get_shape().as_list()
            aug_ver_attrs = tf.concat([ver_rgb, ver_mask], axis=2)
            attrs, _ = rasterize_clip_space(
                clip_xyzw, aug_ver_attrs, tri, imageW, imageH, -1.0
            )

            # Have shading
            diffuse_image = tf.reshape(
                attrs[:, :, :, :3], [batch_size, imageH, imageW, 3]
            )
            alphas = tf.reshape(attrs[:, :, :, 3:], [batch_size, imageH, imageW, 1])
            rgb_images, shading_image = Shader.sh_shader(
                norm_image, alphas, ori_img, para_illum, diffuse_image
            )
            ori_img_remove_shading = ori_img / shading_image

            diffuse_image = tf.clip_by_value(diffuse_image, 0, 1)
            rgb_images = tf.clip_by_value(rgb_images, 0, 1)
            attrs_image = tf.clip_by_value(alphas, 0, 1)
            ori_img_remove_shading = tf.clip_by_value(ori_img_remove_shading, 0, 1)

            render_image = rgb_images * attrs_image + ori_img * (1 - attrs_image)

        return render_image, attrs_image, ori_img_remove_shading

    # @staticmethod
    # def render_fake_view(ori_img, norm_image, alphas, imageH, imageW, uv_rgb, para_illum, batch_vt_indices):
    #     batch_size,_,_,_ = ori_img.get_shape().as_list()

    #     diffuse_image = tf.reshape(tf.gather_nd(uv_rgb,batch_vt_indices),[batch_size,imageH,imageW,3])

    #     # Have shading
    #     para_light = para_illum
    #     background = ori_img
    #     rgb_images, shading_image = Shader.sh_shader(norm_image, alphas, background, para_light, diffuse_image)

    #     diffuse_image = tf.clip_by_value(diffuse_image,0,1)
    #     rgb_images = tf.clip_by_value(rgb_images,0,1)
    #     uv_attrs_image = tf.clip_by_value(alphas,0,1)
    #     shading_image = tf.clip_by_value(shading_image,0,1)

    #     render_image = rgb_images
    #     render_image = render_image * uv_attrs_image + ori_img * (1 - uv_attrs_image)

    #     return render_image

    @staticmethod
    def tf_rotationVector_2_trans(pose6, project_type="Pers", scale=1.0):
        """
        :param:
            pose6: [B, 6, 1], pose paramters

        :return:
            rr :[B, 3, 3] , tt:[B, 3, 1]
        """
        batch_size = pose6.shape[0]
        a, b, c, tx, ty, sth = tf.split(pose6, 6, axis=1)  # B x 1 x 1
        a = a + 0.00001
        b = b + 0.00001
        c = c + 0.00001
        theta = tf.sqrt(tf.multiply(a, a) + tf.multiply(b, b) + tf.multiply(c, c))
        zeros = tf.zeros_like(theta)
        ones = tf.ones_like(theta)

        def tf_Rodrigues(a, b, c, theta):
            kx = a / theta
            ky = b / theta
            kz = c / theta
            n = tf.concat([kx, ky, kz], axis=1)  # B x 3 x 1

            sin_theta = tf.sin(theta)  # B x 1 x 1
            cos_theta = tf.cos(theta)  # B x 1 x 1
            zeros = tf.zeros_like(sin_theta)
            ones = tf.ones_like(sin_theta)
            n_hat = tf.concat(
                [zeros, -1 * kz, ky, kz, zeros, -1 * kx, -1 * ky, kx, zeros], axis=2
            )  # B x 1 x 9
            n_hat = tf.reshape(n_hat, [-1, 3, 3])  # B x 3 x 3
            I = tf.eye(3, 3, batch_shape=[batch_size])  # B x 3 x 3

            # rr0 = cos_theta * I + (1 - cos_theta) * (n * tf.transpose(n)) + sin_theta * n_hat

            cos_theta = tf.tile(cos_theta, [1, 3, 3])  # B x 3 x 3
            sin_theta = tf.tile(sin_theta, [1, 3, 3])  # B x 3 x 3

            rr0 = (
                tf.multiply(cos_theta, I)
                + tf.multiply((1 - cos_theta), tf.matmul(n, tf.transpose(n, [0, 2, 1])))
                + tf.multiply(sin_theta, n_hat)
            )

            return rr0

        if project_type == "Pers":
            rr = tf_Rodrigues(a, b, c, theta)  # B x 3 x 3
            tt = tf.concat([tx, ty, sth], axis=1)  # B x 3 x 1
        else:
            print("Orth")
            rr = tf_Rodrigues(a, b, c, theta) * tf.abs(sth)  # B x 3 x 3
            tt = tf.concat([tx, ty, ones * 50], axis=1)  # B x 3 x 1

        T = tf.concat([rr, tt], axis=2) * scale  # B * 3 * 4
        w = tf.concat([zeros, zeros, zeros, ones], axis=2)  # B * 1 * 4
        T = tf.concat([T, w], axis=1)  # B,4,4

        T = tf.transpose(T, [0, 2, 1])

        return T

    @staticmethod
    def gen_fix_multi_pose(batch_size, project_type):
        """
        generate frontal, left side and right side pose for each sample
        """
        if project_type == "Pers":
            frontal = tf.constant([0.0, np.pi, 0.0, 0.0, 0.0, 50.0], tf.float32)
            left = tf.constant(
                [0.0, np.pi + np.pi / 8, 0.0, 0.0, 0.0, 50.0], tf.float32
            )
            right = tf.constant(
                [0.0, np.pi - np.pi / 8, 0.0, 0.0, 0.0, 50.0], tf.float32
            )
        else:
            frontal = tf.constant([0.0, np.pi, 0.0, 0.0, 0.0, 10.0], tf.float32)
            left = tf.constant(
                [0.0, np.pi + np.pi / 8, 0.0, 0.0, 0.0, 10.0], tf.float32
            )
            right = tf.constant(
                [0.0, np.pi - np.pi / 8, 0.0, 0.0, 0.0, 10.0], tf.float32
            )

        frontal = tf.stack([frontal] * batch_size, axis=0)
        frontal = tf.reshape(frontal, [batch_size, 6, 1])

        left = tf.stack([left] * batch_size, axis=0)
        left = tf.reshape(left, [batch_size, 6, 1])

        right = tf.stack([right] * batch_size, axis=0)
        right = tf.reshape(right, [batch_size, 6, 1])

        return frontal, left, right

    @staticmethod
    def gen_fix_multi_light(batch_size):
        """
        generate frontal, left and right side illumination parameters.
        """
        frontal = tf.reshape(
            tf.constant([0.8, -1.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], tf.float32),
            [1, 9],
        )
        frontal = tf.tile(frontal, [batch_size, 3])

        left = tf.reshape(
            tf.constant([0.6, -0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], tf.float32),
            [1, 9],
        )
        left = tf.tile(left, [batch_size, 3])

        right = tf.reshape(
            tf.constant([0.6, -0.8, -0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], tf.float32),
            [1, 9],
        )
        right = tf.tile(right, [batch_size, 3])

        return frontal, left, right
