#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import Optional
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
# from pytorch3d.transforms import quaternion_multiply
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import tinycudann as tcnn

class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        super().__init__()

        self.active_sh_degree = 0
        self.max_sh_degree = 0 #sh_degree  

        
        self._features_dc_mesh = torch.empty(0)
        self._features_rest_mesh = torch.empty(0)
        self._scaling_mesh = torch.empty(0)


        self._xyz = torch.empty(0)
        # self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        #新增 关于颜色
        self.recolor = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447,
            },
        )

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 3 
            },
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=(self.direction_encoding.n_output_dims+self.recolor.n_output_dims),
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
        # for binding GaussianModel to a mesh
        self.face_center = None
        self.face_scaling = None
        self.face_orien_mat = None
        self.face_orien_quat = None
        # self.binding = None  # gaussian index to face index
        # self.binding_counter = None  # number of points bound to each face
        self.timestep = None  # the current timestep
        self.num_timesteps = 1  # required by viewers

        self.register_buffer("binding", None)
        self.register_buffer("binding_counter", None)

    # def capture(self):
    #     return (
    #         self.active_sh_degree,
    #         self._xyz,
    #         self._features_dc,
    #         self._features_rest,
    #         self._scaling,
    #         self._rotation,
    #         self._opacity,
    #         self.binding,
    #         self.binding_counter,
    #         self.max_radii2D,
    #         self.xyz_gradient_accum,
    #         self.denom,
    #         self.optimizer.state_dict(),
    #         self.spatial_lr_scale,
    #     )
    
    # def restore(self, model_args, training_args):
    #     (self.active_sh_degree, 
    #     self._xyz, 
    #     self._features_dc, 
    #     self._features_rest,
    #     self._scaling, 
    #     self._rotation, 
    #     self._opacity,
    #     self.binding,
    #     self.binding_counter,
    #     self.max_radii2D, 
    #     xyz_gradient_accum, 
    #     denom,
    #     opt_dict, 
    #     self.spatial_lr_scale) = model_args
    #     self.training_setup(training_args)
    #     self.xyz_gradient_accum = xyz_gradient_accum
    #     self.denom = denom
    #     self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling_mesh(self): #获取每个高斯点（Gaussian）的最终缩放因子 scaling
        return self.scaling_activation(self._scaling_mesh).repeat(1,3) # self._scaling 是可学习的原始尺度参数
    
    @property
    def get_scaling(self): #获取每个高斯点（Gaussian）的最终缩放因子 scaling
        if self.binding is None:
            return self.scaling_activation(self._scaling) # self._scaling 是可学习的原始尺度参数
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_scaling is None:
                self.select_mesh_by_timestep(0) # 初始化每个 mesh 面片（或顶点）的尺度因子

            scaling = self.scaling_activation(self._scaling) # scaling：每个 Gaussian 独立学习得到的尺度
            return scaling * self.face_scaling[self.binding] #   Gaussian 的尺度 = 自身可学习尺度 × mesh 局部尺度
    
    @property
    def get_rotation(self): #获取每个高斯点（Gaussian）的最终旋转四元数
        if self.binding is None:
            return self.rotation_activation(self._rotation) # self._rotation 是可学习的原始旋转参数
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_orien_quat is None:
                self.select_mesh_by_timestep(0) #初始化

            # always need to normalize the rotation quaternions before chaining them
            rot = self.rotation_activation(self._rotation) # 1. Gaussian 自身的局部旋转（可学习）
            face_orien_quat = self.rotation_activation(self.face_orien_quat[self.binding]) # 2. mesh 提供的局部旋转（face orientation）
            return quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot)))   # 3. 四元数链式相乘（mesh 旋转 ∘ Gaussian 局部旋转）
            # return quaternion_multiply(face_orien_quat, rot)  # pytorch3d
    
    @property
    def get_xyz(self): #获取每个高斯点（Gaussian）的最终三维位置（世界坐标系）
        if self.binding is None:
            return self._xyz
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_center is None:
                self.select_mesh_by_timestep(0)
            
            xyz = torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1) # 1. 从 mesh 局部坐标系旋转到世界坐标系
            return xyz * self.face_scaling[self.binding] + self.face_center[self.binding] # 2. 应用 mesh 局部尺度并平移到世界坐标
    
    @property
    def get_xyz_cano(self): #获取每个高斯点（Gaussian）的最终三维位置（世界坐标系）   
        if self.binding is None:
            return self._xyz
        else:
            xyz = torch.bmm(self.face_orien_mat_cano[self.binding], self._xyz[..., None]).squeeze(-1) # 1. 从 mesh 局部坐标系旋转到世界坐标系
            return xyz * self.face_scaling_cano[self.binding] + self.face_center_cano[self.binding] # 2. 应用 mesh 局部尺度并平移到世界坐标
    # @property
    # def get_features(self): # 获取每个高斯点（Gaussian）的最终外观特征（SH / color features）
    #     features_dc = self._features_dc
    #     features_rest = self._features_rest
    #     return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_mesh(self): # 获取每个高斯点（Gaussian）的最终外观特征（SH / color features）
        features_dc = self._features_dc_mesh
        features_rest = self._features_rest_mesh
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def select_mesh_by_timestep(self, timestep):
        raise NotImplementedError #直接调用会报错，需要开发者覆盖此方法，这里在flame_gaussian_model.py文件里重写了

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree: # 如果当前激活的 SH 阶数还没有达到允许的最大阶数
            self.active_sh_degree += 1   # 将当前 SH 阶数加 1
    
    def create_from_pcd_mesh(self, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale # 1. 保存空间学习率缩放
        
        
        num_pts = self.flame_model.vertex_num_upsampled 
       
        # fused_color = torch.tensor(np.random.random((num_pts, 3)) / 255.0).float().cuda() # 初始化随机颜色 (N, 3)，范围 [0, 1)
        fused_color = torch.full((num_pts, 3),0.5,dtype=torch.float32,device="cuda") #常数中性灰
 
        # #render需要改
        # self.max_sh_degree = 3
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # 4. 初始化 features 张量
        features = torch.zeros((fused_color.shape[0], 3, 1)).float().cuda() # 4. 初始化 features 张量
        features[:, :3, 0 ] = fused_color # DC 分量（l=0）初始化为点云颜色
        features[:, 3:, 1:] = 0.0 # 高阶分量初始化为 0
       
        print("Number of Mesh Verts at initialisation: ", num_pts)  # 打印初始化点数

        # if self.binding is None: # 6. 初始化尺度 # 未绑定 mesh → 根据点间距离初始化尺度
        dist2 = torch.clamp_min(distCUDA2(self.xyz_cano), 0.0000001) # 防止 log(0)
        scales = torch.log(torch.sqrt(dist2))[...,None]
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)  # 每个轴独立

        # scales = torch.log(torch.full((self.get_xyz.shape[0], 3), 0.005, device="cuda"))
        # scales = torch.log(torch.full((num_pts, 1), 0.05, device="cuda"))
        # scales = torch.log(torch.ones((self.get_xyz.shape[0], 3), device="cuda")) # 已绑定 mesh → 初始尺度为 log(1)
        # rots = torch.zeros((num_pts, 4), device="cuda") # 7. 初始化旋转
        # rots[:, 0] = 1 # 单位四元数 (1, 0, 0, 0)
        
        # self._rgb_mesh = nn.Parameter(fused_color.contiguous().requires_grad_(True))
        self._features_dc_mesh = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_mesh = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_mesh = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 10. 记录 2D 投影半径（初始化为 0）

    def create_from_pcd(self, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale # 1. 保存空间学习率缩放
        
        num_pts = self.binding.shape[0] 
        fused_point_cloud = torch.zeros((num_pts, 3)).float().cuda() # 初始化点云位置全为零 (N, 3)
        fused_color = torch.tensor(np.random.random((num_pts, 3)) / 255.0).float().cuda() # 初始化随机颜色 (N, 3)，范围 [0, 1)
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # 4. 初始化 features 张量
        features[:, :3, 0 ] = fused_color # DC 分量（l=0）初始化为点云颜色
        features[:, 3:, 1:] = 0.0 # 高阶分量初始化为 0
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) # 5. 将点云 / features 转为可训练参数
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # DC 分量
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))  # 高阶分量
        print("Number of points at initialisation: ", self.get_xyz.shape[0])  # 打印初始化点数

        scales = torch.log(torch.ones((self.get_xyz.shape[0], 3), device="cuda")) # 已绑定 mesh → 初始尺度为 log(1)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda") # 7. 初始化旋转
        rots[:, 0] = 1 # 单位四元数 (1, 0, 0, 0)

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # 8. 初始化透明度
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # 9. 将所有参数注册为可训练 nn.Parameter
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 10. 记录 2D 投影半径（初始化为 0）

    # def training_setup(self, training_args):
    #     self.percent_dense = training_args.percent_dense # 1. 保存高斯点稠密度比例
    #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 2. 初始化梯度累积辅助变量 # xyz_gradient_accum: 用于累积每个 Gaussian 的位置梯度
    #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # denom: 累积次数计数器，防止稀疏采样导致梯度过小

    #     l = [   # 3. 定义优化器参数组
    #         {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
    #         {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
    #         {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
    #         {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
    #         {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
    #         {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
    #     ]

    #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)  # 4. 创建 Adam 优化器
    #     self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,  #   - position_lr_init * spatial_lr_scale: 初始学习率
    #                                                 lr_final=training_args.position_lr_final*self.spatial_lr_scale, #   - position_lr_final * spatial_lr_scale: 最终学习率
    #                                                 lr_delay_mult=training_args.position_lr_delay_mult, #   - lr_delay_mult: 延迟因子，训练初期保持低 lr
    #                                                 max_steps=training_args.position_lr_max_steps)  # 5. 创建 xyz 学习率调度器（指数衰减）  #   - max_steps: 学习率衰减到最终值的最大步数

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:  # 遍历优化器的参数组
            if param_group["name"] == "xyz":   # 仅更新位置参数 _xyz 的学习率
                lr = self.xyz_scheduler_args(iteration) # 通过指数衰减函数获取当前步的学习率
                param_group['lr'] = lr  # 更新优化器中该参数组的学习率
                return lr  # 返回当前学习率，方便日志记录 / 调试

    # def construct_list_of_attributes(self):
    #     l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    #     # All channels except the 3 DC
    #     for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
    #         l.append('f_dc_{}'.format(i))
    #     for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
    #         l.append('f_rest_{}'.format(i))
    #     l.append('opacity')
    #     for i in range(self._scaling.shape[1]):
    #         l.append('scale_{}'.format(i))
    #     for i in range(self._rotation.shape[1]):
    #         l.append('rot_{}'.format(i))
    #     if self.binding is not None:
    #         for i in range(1):
    #             l.append('binding_{}'.format(i))
    #     return l

    # def save_ply(self, path):
    #     mkdir_p(os.path.dirname(path))

    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

    #     if self.binding is not None:
    #         binding = self.binding.detach().cpu().numpy()
    #         attributes = np.concatenate((attributes, binding[:, None]), axis=1)

    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)

    def reset_opacity(self): #重置高斯点的不透明度（opacity）参数
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)) # 1. 限制最小值，并从 sigmoid 空间映射回参数空间
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity") # 2. 将更新后的张量替换到优化器参数中
        self._opacity = optimizable_tensors["opacity"] # 3. 更新 _opacity 参数

    def load_ply(self, path, **kwargs):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        # optional fields
        binding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("binding")]
        if len(binding_names) > 0:
            binding_names = sorted(binding_names, key = lambda x: int(x.split('_')[-1]))
            binding = np.zeros((xyz.shape[0], len(binding_names)), dtype=np.int32)
            for idx, attr_name in enumerate(binding_names):
                binding[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self.binding = torch.tensor(binding, dtype=torch.int32, device="cuda").squeeze(-1)

    def replace_tensor_to_optimizer(self, tensor, name): # 用新的张量替换 optimizer 中指定参数组的原参数，并重置 Adam 的动量状态
        optimizable_tensors = {}
        for group in self.optimizer.param_groups: # 遍历 optimizer 参数组
            if group["name"] == name: # 只处理指定名称的参数组
                stored_state = self.optimizer.state.get(group['params'][0], None)  # 1. 获取原参数的 Adam state
                stored_state["exp_avg"] = torch.zeros_like(tensor)  # 2. 重置动量  # Adam 保存一阶动量 exp_avg 和二阶动量 exp_avg_sq
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor) # 因为参数被替换，需要用 zeros_like 重置

                del self.optimizer.state[group['params'][0]]  # 3. 删除旧参数 state
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))   # 4. 用新 tensor 替换 optimizer 参数
                self.optimizer.state[group['params'][0]] = stored_state   # 5. 将重置后的 state 绑定到新参数

                optimizable_tensors[group["name"]] = group["params"][0]  # 6. 保存到返回字典
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}   # 用于返回裁剪后的参数张量
        for group in self.optimizer.param_groups:  # 遍历 optimizer 中的所有 parameter group
            # rule out parameters that are not properties of gaussians
            if len(group["params"]) != 1 or group["params"][0].shape[0] != mask.shape[0]:  # 2. 取出该参数的 optimizer state（Adam）
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]   # 3. 裁剪 Adam 的动量状态
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]  # 删除旧 parameter 对应的 state
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))  # 4. 创建新的 Parameter（裁剪后）
                self.optimizer.state[group['params'][0]] = stored_state # 将裁剪后的 state 重新绑定到新 parameter

                optimizable_tensors[group["name"]] = group["params"][0] # 保存到返回字典
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))  # 5. 没有 optimizer state（如 SGD / 刚初始化）
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        if self.binding is not None:
            # make sure each face is bound to at least one point after pruning
            binding_to_prune = self.binding[mask]  # 取出“将要被 prune 的点”对应的 binding id
            counter_prune = torch.zeros_like(self.binding_counter) # - 用于统计“每个 binding id 将被删掉多少个点”
            counter_prune.scatter_add_(0, binding_to_prune, torch.ones_like(binding_to_prune, dtype=torch.int32, device="cuda"))# 对应 binding id 的位置 +1
            mask_redundant = (self.binding_counter - counter_prune) > 0 # 2. 判断哪些 prune 是“安全的”  # >0 这个 binding id 至少还会剩 1 个点
            mask[mask.clone()] = mask_redundant[binding_to_prune] # 3. 修正 mask（防止误删最后一个点）

        valid_points_mask = ~mask # 4. 得到最终“保留点”的 mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask) # 5. 从 optimizer 中真正删除这些点

        self._xyz = optimizable_tensors["xyz"]  # 6. 更新类内部的参数引用
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]  # 7. 同步裁剪所有辅助缓存  # 梯度累计缓存

        self.denom = self.denom[valid_points_mask] # 梯度归一化分母
        self.max_radii2D = self.max_radii2D[valid_points_mask] # screen-space 最大半径缓存

        if self.binding is not None:  # 8. 更新 binding 及其计数（若存在）
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            self.binding_counter.scatter_add_(0, self.binding[mask], -torch.ones_like(self.binding[mask], dtype=torch.int32, device="cuda"))   # 对被删除的点，其 binding id 的计数 -1
            self.binding = self.binding[valid_points_mask]    # 最后，对 binding 本身也做 mask

    def cat_tensors_to_optimizer(self, tensors_dict): # 将新的张量（tensors_dict）拼接到现有 optimizer 管理的参数中
        optimizable_tensors = {}
        for group in self.optimizer.param_groups: # 遍历 optimizer 的参数组
            # rule out parameters that are not properties of gaussians
            if group["name"] not in tensors_dict:  # 1. 跳过不在 tensors_dict 的参数组
                continue
            
            assert len(group["params"]) == 1 # 确保每组参数只有一个 tensor
            extension_tensor = tensors_dict[group["name"]] # 待拼接的新 tensor
            stored_state = self.optimizer.state.get(group['params'][0], None)  # 获取原 Adam 状态（momentum, rms 等）
            if stored_state is not None:  # 2. 如果已有优化器状态，需要扩展动量张量

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)  # 扩展一阶动量（exp_avg）
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)  # 扩展二阶动量（exp_avg_sq）

                del self.optimizer.state[group['params'][0]]  # 删除原来的 state，以便注册新参数
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))   # 拼接原参数与新参数，并注册为可训练 nn.Parameter
                self.optimizer.state[group['params'][0]] = stored_state  # 将扩展后的 state 重新绑定到新参数

                optimizable_tensors[group["name"]] = group["params"][0]   # 保存到返回字典
            else: # 3. 如果没有原优化器状态，直接拼接
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors # 4. 返回更新后的可训练张量字典

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,  # 1. 将新高斯点的各类参数打包成字典
        # "f_dc": new_features_dc, # d 中的 key 必须与 optimizer 中 parameter group 的 name 一致
        # "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)   # 2. 将新参数拼接到 optimizer 中（最关键的一步）
        self._xyz = optimizable_tensors["xyz"] # 更新高斯点坐标（现在点数 = 旧点数 + 新点数）
        # self._features_dc = optimizable_tensors["f_dc"] # 更新 DC 球谐特征
        # self._features_rest = optimizable_tensors["f_rest"] # 更新高阶球谐特征
        self._opacity = optimizable_tensors["opacity"] # 更新不透明度
        self._scaling = optimizable_tensors["scaling"] # 更新缩放参数
        self._rotation = optimizable_tensors["rotation"] # 更新旋转参数

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")   # 4. 重置与“点数量相关”的缓存张量  # 由于点数发生了变化，必须重新初始化
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # - 通常作为 xyz_gradient_accum 的归一化分母
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 同样必须和当前点数严格一致

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]  # n_init_points：densify 之前的点数
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda") # 2. 对梯度进行 padding（对齐点数）# padded_grad 的长度与当前高斯点数量一致
        padded_grad[:grads.shape[0]] = grads.squeeze()  # 将已有梯度拷贝到 padded_grad 的前半部分
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)  # 3. 根据梯度大小筛选候选 split 点 # 梯度大于阈值的点才有 split 的价值
        selected_pts_mask = torch.logical_and(selected_pts_mask,     # 再加一个约束：尺度必须“足够大”
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)  # 4. 为 split 采样局部扰动（核心几何步骤）
        means =torch.zeros((stds.size(0), 3),device="cuda") # 均值为 0，表示在局部坐标系中采样
        samples = torch.normal(mean=means, std=stds) # 从高斯分布中采样局部偏移
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)  # 5. 将局部采样旋转到世界坐标系
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)   # 使用 batch matrix multiplication
        if self.binding is not None: # 6. 计算新高斯点的尺度（缩小）
            selected_scaling = self.get_scaling[selected_pts_mask]
            face_scaling = self.face_scaling[self.binding[selected_pts_mask]]
            new_scaling = self.scaling_inverse_activation((selected_scaling / face_scaling).repeat(N,1) / (0.8*N))
        else: # 无 binding 的普通情况
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)  # 7. 复制其余属性（外观参数）
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        if self.binding is not None: # 8. 处理 binding（若存在）
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            new_binding = self.binding[selected_pts_mask].repeat(N) # 新子点继承父点的绑定关系
            self.binding = torch.cat((self.binding, new_binding))  # 拼接到 binding 列表
            self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device="cuda")) # 更新每个 binding 的点数统计

        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation) # 9. 将新 split 出来的点加入模型

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))  # 10. prune 掉原来的“大高斯点”
        self.prune_points(prune_filter)  # 真正执行裁剪

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) # 计算每个高斯点梯度的 L2 范数（沿最后一个维度）  # 表示每个高斯点的“重要程度”
        selected_pts_mask = torch.logical_and(selected_pts_mask,  # 2. 再加一个约束：高斯点的尺度不能太大
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # 3. 从原始高斯属性中取出被选中的点（clone）
        new_xyz = self._xyz[selected_pts_mask]  # 取出需要复制的高斯点坐标
        # new_features_dc = self._features_dc[selected_pts_mask] # 取出 DC（0 阶）球谐系数，表示基础颜色
        # new_features_rest = self._features_rest[selected_pts_mask] # 取出高阶球谐系数（细节颜色）
        new_opacities = self._opacity[selected_pts_mask] # 取出透明度参数
        new_scaling = self._scaling[selected_pts_mask]  # 取出尺度参数（xyz 三个方向）
        new_rotation = self._rotation[selected_pts_mask]  # 取出旋转参数（通常是四元数）
        if self.binding is not None: # 4. 若存在 binding（绑定到 mesh / bone / triangle）
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            new_binding = self.binding[selected_pts_mask]  # 取出被复制点对应的 binding 索引
            self.binding = torch.cat((self.binding, new_binding)) # 将新的 binding 拼接到原 binding 列表中
            self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device="cuda")) # 更新 binding_counter
        
        self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation)  # 5. 将新复制的高斯点真正加入模型参数中

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom # 1. 计算累积梯度 # xyz_gradient_accum: 累积的位置梯度 # denom: 累积的梯度计数（用于平均） # grads: 平均梯度，形状 (N_gaussians, 3)
        grads[grads.isnan()] = 0.0    # 处理NaN值：将NaN梯度设为0
        # 2. 基于梯度的密度增加操作
        self.densify_and_clone(grads, max_grad, extent)# 2.1 克隆操作（densify_and_clone）
        self.densify_and_split(grads, max_grad, extent)# 2.2 分裂操作（densify_and_split）

        prune_mask = (self.get_opacity < min_opacity).squeeze()# 3.1 基础修剪：基于不透明度 # 结果：prune_mask[i] = True 表示第i个高斯点需要修剪
        if max_screen_size: # 4. 额外的修剪条件（如果提供了max_screen_size）
            big_points_vs = self.max_radii2D > max_screen_size # 4.1 屏幕空间过大判断
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent # 4.2 世界空间过大判断
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws) # 4.3 合并所有修剪条件
        self.prune_points(prune_mask) # 5. 执行修剪操作

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x