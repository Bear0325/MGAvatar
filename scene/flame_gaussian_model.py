# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

from nets.loss import RGBLoss, SSIM, LaplacianReg
from flame_model.lbs import lbs
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from utils.general_utils import get_expon_lr_func

from utils.uv_utils import PositionMapGenerator, load_uv_region_masks
from utils.mesh_sampling import (
    reweight_uvcoords_by_barycoords,
    hybrid_sampling_by_binding,
)

class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100):
        super().__init__(sh_degree)
        self.stage2_is_on = False
        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None

        # 1. 三平面表示参数（用于几何和外观编码） # 三平面表示：三个正交的特征平面（XY、XZ、YZ）
        self.triplane_head = nn.Parameter(torch.zeros((3, 32, 128, 128), device='cuda', dtype=torch.float32))
        #第一阶段网络
        # 2. 几何网络：从三平面特征预测几何属性
        self.geo_net = self.make_linear_layers([3 * 32, 128, 128, 128],use_gn=True).cuda()
        self.mean_offset_net = self.make_linear_layers([128, 3], relu_final=False).cuda() # 均值偏移
        # # 3. 几何偏移网络：考虑姿态依赖的几何变化 # 输入：三平面特征 + 关节旋转信息 (smpl_x.joint_num-1)*6 (旋转矩阵的6D表示)
        self.geo_offset_net = self.make_linear_layers([3 * 32+(self.flame_model.joint_num-1)*6, 128, 128, 128], use_gn=True).cuda()
        self.mean_offset_offset_net = self.make_linear_layers([128, 3], relu_final=False).cuda() # 额外的均值偏移

        # self.rgb_loss = RGBLoss()
        # self.ssim = SSIM()
        # self.lpips = LPIPS()
        self.lap_reg = LaplacianReg(self.flame_model.vertex_num_upsampled, self.flame_model.face_upsampled)

    def init(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        """
        初始化模型，准备SMPLX相关的缓冲区和上采样网格
        """
        # 1. 获取中性姿态的人体网格顶点
        # upsample mesh and other assets
        self.load_meshes(train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes) 
        verts_cano = self.verts_cano[0].cuda()
        xyz = self.flame_model.upsample_mesh(verts_cano)
        # self.xyz_cano = xyz
       
        # 2. 获取SMPLX的蒙皮权重和各种变换矩阵
        skinning_weight = self.flame_model.lbs_weights.float() # 线性混合蒙皮权重
        pose_dirs = self.flame_model.posedirs.permute(1,0).reshape(self.flame_model.vertex_num,3*(self.flame_model.joint_num-1)*9) # 姿态依赖的形变
        expr_dirs = self.flame_model.exprdirs.view(self.flame_model.vertex_num,3*self.flame_model.n_expr_params) # 表情依赖的形变
        shape_dirs = self.flame_model.shapedirs.view(self.flame_model.vertex_num,3*(self.flame_model.n_shape_params + self.flame_model.n_expr_params))
        J_regressor = self.flame_model.J_regressor.T
        # 3. 创建身体部位掩码（右手、左手、面部、表情区域）
        is_face_expr = torch.zeros((self.flame_model.vertex_num,1)).float().cuda()
        is_face_expr[self.flame_model.expr_vertex_idx] = 1.0  # 设置对应顶点的掩码值
        # is_neck[self.flame_model.neck_idx] = 1.0
        # 5. 上采样网格和相关属性到更高分辨率 # 使用虚拟顶点进行上采样
        # _, skinning_weight, pose_dirs, expr_dirs, is_face, is_face_expr = self.flame_model.upsample_mesh(torch.ones((self.flame_model.vertex_num,3)).float().cuda(), [skinning_weight, pose_dirs, expr_dirs, is_face, is_face_expr]) # upsample with dummy vertex
        # _, skinning_weight, pose_dirs, expr_dirs, J_regressor, is_face, is_face_expr = self.flame_model.upsample_mesh(torch.ones((self.flame_model.vertex_num,3)).float().cuda(), [skinning_weight, pose_dirs, expr_dirs, J_regressor, is_face, is_face_expr])
        _, skinning_weight, pose_dirs, expr_dirs, shape_dirs, J_regressor, is_face_expr = self.flame_model.upsample_mesh(torch.ones((self.flame_model.vertex_num,3)).float().cuda(), [skinning_weight, pose_dirs, expr_dirs, shape_dirs, J_regressor, is_face_expr])
        
        # shapedirs = self.flame_model.upsample_mesh(torch.ones((self.flame_model.vertex_num,3)).float().cuda(),
        # 6. 重新组织上采样后的数据
        pose_dirs = pose_dirs.reshape(self.flame_model.vertex_num_upsampled*3,(self.flame_model.joint_num-1)*9).permute(1,0) 
        expr_dirs = expr_dirs.view(self.flame_model.vertex_num_upsampled,3,self.flame_model.n_expr_params)
        shape_dirs = shape_dirs.view(self.flame_model.vertex_num_upsampled,3,(self.flame_model.n_shape_params + self.flame_model.n_expr_params))
        J_regressor = J_regressor.T
        row_sums = J_regressor.sum(dim=1, keepdim=True)
        J_regressor = J_regressor / row_sums
        is_face_expr = is_face_expr[:,0] > 0  # 将掩码转换为布尔值
        # 7. 注册缓冲区（不参与梯度计算但需要保存的常量）
        self.register_buffer('xyz_cano', xyz)
        self.register_buffer('position', xyz)  # 位置编码网格
        self.register_buffer('skinning_weight', skinning_weight)  # 蒙皮权重
        self.register_buffer('pose_dirs', pose_dirs) # 姿态依赖形变
        self.register_buffer('expr_dirs', expr_dirs)  # 表情依赖形变
        self.register_buffer('shape_dirs', shape_dirs)
        self.register_buffer('J_regressor', J_regressor)
        self.register_buffer('is_face_expr', is_face_expr) # 表情区域掩码
        
        self.create_from_pcd_mesh(self.cameras_extent) 
        
    
    def make_linear_layers(self, feat_dims, relu_final=True, use_gn=False):
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

            # Do not use ReLU for final estimation
            if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
                if use_gn:
                    layers.append(nn.GroupNorm(4, feat_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
    
    def extract_tri_feature(self, point):
    # def extract_tri_feature(self, point, is_face):
        ## 1. triplane features of all vertices
        # normalize coordinates to [-1,1]
        xyz = point
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / 0.25
        y = xyz[:,1] / 0.25
        z = xyz[:,2] / 0.25
        
        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane_head[0,None,:,:,:], xy[None,:,None,:],align_corners=False)[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane_head[1,None,:,:,:], xz[None,:,None,:],align_corners=False)[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane_head[2,None,:,:,:], yz[None,:,None,:],align_corners=False)[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # smpl_x.vertex_num_upsampled, cfg.triplane_shape[0]*3

        return tri_feat

    #后lbs
    def forward(self, timestep, original=False):
        self.timestep = timestep 
        self.update_cano_mesh()
        flame_param = self.flame_param
        verts_cano = self.verts_cano[0].cuda()
        self.template = self.flame_model.upsample_mesh(self.template[0].cuda())
        template = self.flame_model.upsample_mesh(verts_cano)
        self.xyz_cano = template
        # template = self.xyz_cano

        tri_feat = self.extract_tri_feature(self.position) 
        # 3. 获取基础高斯资产
        # get Gaussian assets
        geo_feat = self.geo_net(tri_feat)  # 几何特征
        mean_offset = self.mean_offset_net(geo_feat)  # 高斯均值偏移（相对于中性网格）# mean offset of Gaussians
        mean_3d = template + mean_offset 
        
  # 4. 获取姿态依赖的高斯资产（精炼版本）
        # get pose-dependent Gaussian assets
        mean_offset_offset = self.forward_geo_network(tri_feat, flame_param, timestep)
      
        mean_combined_offset, mean_offset_offset = self.get_mean_offset_offset(flame_param, timestep, mean_offset_offset)  # 均值偏移处理
        mean_3d_refined = mean_3d + mean_combined_offset 
        
        #修正
        # mean_offset_offset = self.get_mean_offset_offset(mean_offset_offset)  # 修正均值偏移处理
        # mean_3d_refined = mean_3d + mean_offset_offset 

        # # 5. 添加SMPLX表情偏移
        # # smplx facial expression offset
        # smplx_expr_offset = (flame_param['expr'][[timestep]].squeeze(0)[None,None,:] * self.expr_dirs).sum(2) # 计算表情引起的顶点偏移：expr参数 × 表情形变基
        # mean_3d = mean_3d + smplx_expr_offset
        # mean_3d_refined = mean_3d_refined + smplx_expr_offset 

        shape = torch.zeros_like(flame_param['shape'][None, ...])    
        betas = torch.cat([shape, flame_param['expr'][[timestep]]], dim=1)
        mean_3d = mean_3d + blend_shapes(betas, self.shape_dirs)
        mean_3d_refined = mean_3d_refined + blend_shapes(betas, self.shape_dirs)

        full_pose = torch.cat([flame_param['rotation'][[timestep]], flame_param['neck_pose'][[timestep]], flame_param['jaw_pose'][[timestep]], flame_param['eyes_pose'][[timestep]]], dim=1) 
        mean_3d, J, mat_rot = lbs( #应用线性混合蒙皮 (LBS) 
            full_pose, #当前姿态
            mean_3d, # 形状+表情+静态偏移后的顶点
            self.pose_dirs, # pose 方向基
            self.J_regressor, # 回归关节位置
            self.flame_model.parents,  # 父子关节关系
            self.skinning_weight, # LBS 权重
            dtype=self.flame_model.dtype,
        )
        mean_3d = mean_3d + flame_param['translation'][[timestep]][:, None, :]

        mean_3d_refined, J, mat_rot = lbs( #应用线性混合蒙皮 (LBS) 
            full_pose, #当前姿态
            mean_3d_refined, # 形状+表情+静态偏移后的顶点
            self.pose_dirs, # pose 方向基
            self.J_regressor, # 回归关节位置
            self.flame_model.parents,  # 父子关节关系
            self.skinning_weight, # LBS 权重
            dtype=self.flame_model.dtype,
        )
        mean_3d_refined = mean_3d_refined + flame_param['translation'][[timestep]][:, None, :] # 应用全局平移 

        mean_3d_all = mean_3d.squeeze(0)
        mean_3d_refined_all = mean_3d_refined.squeeze(0)
        
        rotation_all = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(self.flame_model.vertex_num_upsampled,1,1)) # constant rotation # 旋转：使用单位四元数（无旋转）
        opacity_all = torch.ones((self.flame_model.vertex_num_upsampled,1)).float().cuda() 
        
        assets = { # 基础资产
                'mean_3d': mean_3d_all, # 3D位置
                'opacity': opacity_all,   # 不透明度
                'rotation': rotation_all,   # 旋转（四元数）
                }
        assets_refined = { # 精炼资产
                'mean_3d': mean_3d_refined_all, 
                'opacity': opacity_all,   # 不透明度
                'rotation': rotation_all,   # 旋转（四元数）
                }
        offsets = { # 偏移量（用于正则化损失）
                'mean_offset': mean_offset, # 基础均值偏移
                'mean_offset_offset': mean_offset_offset, # 姿态依赖的均值偏移
                }      
        return assets, assets_refined, offsets    
        
    def forward_geo_network(self, tri_feat, flame_param, timestep):
        """
        FLAME 版 forward_geo_network，用于人头重建

        Args:
            tri_feat: triplane 特征，形状 [vertex_num, tri_feat_dim]
            flame_param: FLAME 参数字典，包括
                'shape', 'expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation'
            flame_vertex_num: 顶点数量
        Returns:
            mean_offset_offset: 高斯的姿态相关平均偏移
            scale_offset: 高斯的姿态相关尺度偏移
        """

        # 提取头部相关 pose
        # rotation = flame_param['rotation'][[timestep]].view(1, 3)
        neck_pose = flame_param['neck_pose'][[timestep]].view(1, 3)
        jaw_pose = flame_param['jaw_pose'][[timestep]].view(1, 3)
        eyes_pose = flame_param['eyes_pose'][[timestep]].view(-1, 3)  # 如果 eyes_pose 是两只眼睛，可以是 2x3

        # 拼接 pose
        pose = torch.cat([neck_pose, jaw_pose, eyes_pose], dim=0)

        # 转为旋转矩阵再转 6D 表示
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose)).view(1,self.flame_model.joint_num-1,6).repeat(self.flame_model.vertex_num_upsampled,1,1) 
        pose = pose.view(self.flame_model.vertex_num_upsampled, (self.flame_model.joint_num-1)*6)
        feat = torch.cat((tri_feat, pose.detach()),1)

        # 前向网络
        geo_offset_feat = self.geo_offset_net(feat)
        mean_offset_offset = self.mean_offset_offset_net(geo_offset_feat)
        # scale_offset = self.scale_offset_net(geo_offset_feat)

        # return mean_offset_offset, scale_offset
        return mean_offset_offset
    
    def get_mean_offset_offset(self, flame_param, timestep, mean_offset_offset):

        # 提取头部相关 pose
        # rotation = flame_param['rotation'][[timestep]].view(1, 3)
        neck_pose = flame_param['neck_pose'][[timestep]].view(1, 3)
        jaw_pose = flame_param['jaw_pose'][[timestep]].view(1, 3)
        eyes_pose = flame_param['eyes_pose'][[timestep]].view(-1, 3)  # 如果 eyes_pose 是两只眼睛，可以是 2x3
        # 拼接 pose
        pose = torch.cat([neck_pose, jaw_pose, eyes_pose], dim=0)

        # smplx pose-dependent vertex offset
        pose = (axis_angle_to_matrix(pose) - torch.eye(3)[None,:,:].float().cuda()).view(1,(self.flame_model.joint_num-1)*9)
        pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(self.flame_model.vertex_num_upsampled,3)

        # combine it with regressed mean_offset_offset
        # for face and hands, use smplx offset
        mask = ((self.is_face_expr) > 0)[:,None].float()
        mean_offset_offset = mean_offset_offset * (1 - mask)
        pose_offset = pose_offset * mask
        output = mean_offset_offset + pose_offset
        return output, mean_offset_offset    

    # #修正
    # def get_mean_offset_offset(self, mean_offset_offset):

    #     mask = ((self.is_face_expr) > 0)[:,None].float()
    #     mean_offset_offset = mean_offset_offset * (1 - mask)
    #     return mean_offset_offset   


    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers #加1是因为，如果最大key是99，就说明有100帧
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}

            self.update_cano_mesh()
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass

    #仅self.shape
    def update_cano_mesh(self): #更新规范空间（canonical space）的网格信息
        flame_param = self.flame_param # 获取当前的FLAME参数

        verts, verts_cano = self.flame_model(  # 调用FLAME模型生成规范空间的网格
            flame_param['shape'][None, ...],  # 形状参数（保持个体特征）
            torch.zeros_like(flame_param['expr'][[0]]).cuda(),   # 表情参数设为0 → 中性表情
            torch.zeros_like(flame_param['rotation'][[0]]).cuda(),    # 旋转设为0 → 正面朝向
            torch.zeros_like(flame_param['neck_pose'][[0]]).cuda(),   # 颈部姿态设为0 → 直立
            torch.zeros_like(flame_param['jaw_pose'][[0]]).cuda(),   # 下巴姿态设为0 → 闭嘴
            torch.zeros_like(flame_param['eyes_pose'][[0]]).cuda(),  # 眼部姿态设为0 → 睁眼
            torch.zeros_like(flame_param['translation'][[0]]).cuda(), # 平移设为0 → 原点位置
            zero_centered_at_root_node=False,  # 不以根节点为中心进行零化
            return_landmarks=False,    # 不返回面部特征点
            return_verts_cano=True,  # 返回规范空间的顶点
            static_offset=flame_param['static_offset'],  # 使用静态偏移量
            # static_offset=self.static_offset,  # 使用静态偏移量
            dynamic_offset=None, # 不使用动态偏移量
        )  # 使用零值参数来获得中性表情和零姿态的状态

        self.verts_cano = verts  # 存储规范空间的顶点坐标

        #保存template
        self.template, _ = self.flame_model(  # 调用FLAME模型生成规范空间的网格
            flame_param['shape'][None, ...],  # 形状参数（保持个体特征）
            torch.zeros_like(flame_param['expr'][[0]]).cuda(),   # 表情参数设为0 → 中性表情
            torch.zeros_like(flame_param['rotation'][[0]]).cuda(),    # 旋转设为0 → 正面朝向
            torch.zeros_like(flame_param['neck_pose'][[0]]).cuda(),   # 颈部姿态设为0 → 直立
            torch.zeros_like(flame_param['jaw_pose'][[0]]).cuda(),   # 下巴姿态设为0 → 闭嘴
            torch.zeros_like(flame_param['eyes_pose'][[0]]).cuda(),  # 眼部姿态设为0 → 睁眼
            torch.zeros_like(flame_param['translation'][[0]]).cuda(), # 平移设为0 → 原点位置
            zero_centered_at_root_node=False,  # 不以根节点为中心进行零化
            return_landmarks=False,    # 不返回面部特征点
            return_verts_cano=True,  # 返回规范空间的顶点
            static_offset=torch.zeros([5143, 3]).cuda(),  # 使用静态偏移量
            # static_offset=self.static_offset,  # 使用静态偏移量
            dynamic_offset=None, # 不使用动态偏移量
        )  # 使用零值参数来获得中性表情和零姿态的状态
        
    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

 
    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param

        tri_feat_mesh = self.extract_tri_feature(self.position)
        mean_offset_offset_mesh = self.forward_geo_network(tri_feat_mesh, flame_param, timestep)
        # scale, scale_refined = torch.exp(scale).repeat(1,3), torch.exp(scale+scale_offset).repeat(1,3) # 尺度处理：从对数空间转换到实际尺度，并复制到xyz三个维度
        
        mean_combined_offset_mesh, mean_offset_offset_mesh = self.get_mean_offset_offset(flame_param, timestep, mean_offset_offset_mesh)  # 均值偏移处理
        mean_3d_refined_mesh = self.xyz_cano + mean_combined_offset_mesh

        # mean_offset_offset_mesh = self.get_mean_offset_offset(mean_offset_offset_mesh)  # 修正均值偏移处理
        # mean_3d_refined_mesh = self.xyz_cano + mean_offset_offset_mesh 

        with torch.no_grad():
            self.flame_model.v_template.copy_(mean_3d_refined_mesh[:self.flame_model.vertex_num, :])

        shape = torch.zeros_like(flame_param['shape'])
        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep_render(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param

        tri_feat_mesh = self.extract_tri_feature(self.position)
        mean_offset_offset_mesh = self.forward_geo_network(tri_feat_mesh, flame_param, timestep)
     
        mean_combined_offset_mesh, mean_offset_offset_mesh = self.get_mean_offset_offset(flame_param, timestep, mean_offset_offset_mesh)  # 均值偏移处理
        mean_3d_refined_mesh = self.xyz_cano + mean_combined_offset_mesh
        
        # mean_offset_offset_mesh = self.get_mean_offset_offset(mean_offset_offset_mesh)  # 修正均值偏移处理
        # mean_3d_refined_mesh = self.xyz_cano + mean_offset_offset_mesh 

        with torch.no_grad():
            self.flame_model.v_template.copy_(mean_3d_refined_mesh[:self.flame_model.vertex_num, :])

        shape = torch.zeros_like(flame_param['shape'])
        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)


        shape = torch.zeros_like(flame_param['shape'][None, ...])    
        betas = torch.cat([shape, flame_param['expr'][[timestep]]], dim=1)
        mean_3d_refined_mesh = mean_3d_refined_mesh + blend_shapes(betas, self.shape_dirs)

        full_pose = torch.cat([flame_param['rotation'][[timestep]], flame_param['neck_pose'][[timestep]], flame_param['jaw_pose'][[timestep]], flame_param['eyes_pose'][[timestep]]], dim=1) 
        mean_3d_refined_mesh, J, mat_rot = lbs( #应用线性混合蒙皮 (LBS) 
            full_pose, #当前姿态
            mean_3d_refined_mesh, # 形状+表情+静态偏移后的顶点
            self.pose_dirs, # pose 方向基
            self.J_regressor, # 回归关节位置
            self.flame_model.parents,  # 父子关节关系
            self.skinning_weight, # LBS 权重
            dtype=self.flame_model.dtype,
        )
        mean_3d_refined_mesh = mean_3d_refined_mesh + flame_param['translation'][[timestep]][:, None, :] # 应用全局平移

        self.verts = mean_3d_refined_mesh 
        self.faces = torch.as_tensor(self.flame_model.face_upsampled, device=mean_3d_refined_mesh.device)



    def update_mesh_properties(self, verts, verts_cano):
        # 1. 获取网格面片信息
        faces = self.flame_model.faces  # 面片索引 [F, 3]
        triangles = verts[:, faces] # 每个三角面的顶点坐标 [1, F, 3, 3]

        # 2. 计算每个三角面的中心点
        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0) # face_center = 每个三角面三个顶点坐标的平均值

        # 3. 计算每个三角面的局部坐标系（旋转矩阵）和尺度
        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True) # 计算每个三角面的局部坐标系（旋转矩阵）和尺度
        
        # 4. 将旋转矩阵转换为四元数表示
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ) 
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma  

        # 5. 保存网格用于渲染
        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano  # 6. 保存 canonical 网格用于正则化 

    
    def init_stage2(self):
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()
        
        # verts_cano = self.verts_cano[0].cuda()
        # template = self.flame_model.upsample_mesh(verts_cano)
        template = self.xyz_cano
        tri_feat_mesh = self.extract_tri_feature(self.position) 
        # 3. 获取基础高斯资产
        # get Gaussian assets
        geo_feat_mesh = self.geo_net(tri_feat_mesh)  # 几何特征
        mean_offset_mesh = self.mean_offset_net(geo_feat_mesh)  # 高斯均值偏移（相对于中性网格）# mean offset of Gaussians

        mean_3d_mesh = template + mean_offset_mesh
        # self.create_from_canonical(mean_3d_mesh)
        self.xyz_cano = mean_3d_mesh
        # with torch.no_grad():
        #     self.flame_model.v_template.copy_(mean_3d_mesh[:self.flame_model.vertex_num, :])
        self.get_cano_face_info(mean_3d_mesh[:self.flame_model.vertex_num, :].unsqueeze(0))
        self.create_from_pcd(self.cameras_extent)

    def get_cano_face_info(self, verts):
        # 1. 获取网格面片信息
        faces = self.flame_model.faces  # 面片索引 [F, 3]
        triangles = verts[:, faces] # 每个三角面的顶点坐标 [1, F, 3, 3]

        # 2. 计算每个三角面的中心点
        # position
        self.face_center_cano = triangles.mean(dim=-2).squeeze(0) # face_center = 每个三角面三个顶点坐标的平均值

        # 3. 计算每个三角面的局部坐标系（旋转矩阵）和尺度
        # orientation and scale
        self.face_orien_mat_cano, self.face_scaling_cano = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True) # 计算每个三角面的局部坐标系（旋转矩阵）和尺度

    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def update_learning_rate(self, iteration, training_args):
        
        for param_group in self.optimizer.param_groups:  # 遍历优化器的参数组
            if param_group["name"] == "xyz":   # 仅更新位置参数 _xyz 的学习率
                lr = self.xyz_scheduler_args(iteration) # 通过指数衰减函数获取当前步的学习率
                param_group['lr'] = lr  # 更新优化器中该参数组的学习率
                return lr  # 返回当前学习率，方便日志记录 / 调试
        
            # if 'head' in param_group['name']:
            #     if (iteration > 0.75 * training_args.iteration_stage1) and (iteration <= 0.95 * training_args.iteration_stage1):
            #         param_group['lr'] = training_args.lr / 10
            #         # param_group['lr'] = training_args.lr / 20
            #     elif (iteration > 0.95 * training_args.iteration_stage1):
            #         param_group['lr'] = training_args.lr / 100  

            if 'head' in param_group['name']:
                if (iteration > 0.95 * training_args.iteration_stage1):
                    param_group['lr'] = training_args.lr / 10  


    def training_setup(self, training_args):

        l = [   # 3. 定义优化器参数组
            {'params': [self.triplane_head], 'lr': training_args.lr, 'name': 'triplane_head'},
            {'params': list(self.geo_net.parameters()), 'lr': training_args.lr, 'name': 'geo_net_head'},
            {'params': list(self.mean_offset_net.parameters()), 'lr': training_args.lr, 'name': 'mean_offset_net_head'},
            {'params': list(self.geo_offset_net.parameters()), 'lr': training_args.lr, 'name': 'geo_offset_net_head'},
            {'params': list(self.mean_offset_offset_net.parameters()), 'lr': training_args.lr, 'name': 'mean_offset_offset_net_head'},
            {'params': [self._features_dc_mesh], 'lr': training_args.feature_lr, "name": "f_dc_mesh"},
            {'params': [self._features_rest_mesh], 'lr': training_args.feature_lr / 20.0, "name": "f_rest_mesh"},
            # {'params': [self._rgb_mesh], 'lr': training_args.feature_lr, "name": "rgb_mesh"},
            {'params': [self._scaling_mesh], 'lr': training_args.scaling_lr, "name": "scaling_mesh"},
            # {'params': [self.shape], 'name': 'shape_head', 'lr': training_args.lr},
            # {'params': [self.static_offset], 'name': 'static_offset_head', 'lr': training_args.lr / 100.0}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)  # 4. 创建 Adam 优化器

        if self.not_finetune_flame_params:  # 1. 如果不想微调 FLAME 参数，则直接返回
            return
        # shape
        self.flame_param['shape'].requires_grad = True
        param_shape = {'params': [self.flame_param['shape']], 'lr': training_args.lr, "name": "shape_head"}
        self.optimizer.add_param_group(param_shape)

        # pose  # 2. 设置姿态参数（pose）可训练，并添加到优化器
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [   # 将这些参数放入列表
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}   # 创建 optimizer param group
        self.optimizer.add_param_group(param_pose)   # 添加到优化器

        # translation  # 3. 设置平移参数可训练，并添加到优化器
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression  # 4. 设置表情参数可训练，并添加到优化器
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)

  
    def training_setup_stage2(self, training_args):
        # ===== 1. 冻结 Stage1 网络 =====
        stage1_modules = [
            self.triplane_head,
            self.geo_net,
            self.mean_offset_net,
            self.geo_offset_net,
            self.mean_offset_offset_net,
            self._features_dc_mesh,
            self._features_rest_mesh,
            self._scaling_mesh,
            # self.shape
        ]

        for p in stage1_modules:
            if isinstance(p, torch.nn.Parameter):
                p.requires_grad_(False)
            elif isinstance(p, torch.nn.Module):
                for param in p.parameters():
                    param.requires_grad_(False)

        # ===== 2. 冻结 FLAME 参数=====
        # for k in ['shape', 'rotation', 'neck_pose', 'jaw_pose',
        #         'eyes_pose', 'translation', 'expr']:
        for k in ['shape']:
            if k in self.flame_param:
                self.flame_param[k].requires_grad_(False)

        # ===== 2. 设置 Stage2 优化器 =====
        self.percent_dense = training_args.percent_dense # 1. 保存高斯点稠密度比例
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda") # 2. 初始化梯度累积辅助变量 # xyz_gradient_accum: 用于累积每个 Gaussian 的位置梯度
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")  # denom: 累积次数计数器，防止稀疏采样导致梯度过小

        l = [
            #add_point相关
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)  # 4. 创建 Adam 优化器
        
        if self.not_finetune_flame_params:  # 1. 如果不想微调 FLAME 参数，则直接返回
            return
        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': training_args.lr, "name": "shape_head"}
        # self.optimizer.add_param_group(param_shape)

        # pose  # 2. 设置姿态参数（pose）可训练，并添加到优化器
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [   # 将这些参数放入列表
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}   # 创建 optimizer param group
        self.optimizer.add_param_group(param_pose)   # 添加到优化器

        # translation  # 3. 设置平移参数可训练，并添加到优化器
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression  # 4. 设置表情参数可训练，并添加到优化器
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,  #   - position_lr_init * spatial_lr_scale: 初始学习率
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale, #   - position_lr_final * spatial_lr_scale: 最终学习率
                                                    lr_delay_mult=training_args.position_lr_delay_mult, #   - lr_delay_mult: 延迟因子，训练初期保持低 lr
                                                    max_steps=training_args.position_lr_max_steps)  # 5. 创建 xyz 学习率调度器（指数衰减）  #   - max_steps: 学习率衰减到最终值的最大步数

        other_params = []
        for params in self.recolor.parameters():
            other_params.append(params)
        for params in self.mlp_head.parameters():
            other_params.append(params)

        self.optimizer_net = torch.optim.Adam(other_params, lr=training_args.net_lr, eps=1e-15)
        self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler([ #控制学习率时用
            torch.optim.lr_scheduler.LinearLR(self.optimizer_net, start_factor=0.01, total_iters=100),
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_net,
                milestones=training_args.net_lr_step,
                gamma=0.33,
            ),
        ])

    def capture(self):
        cap = (
            self.state_dict(),
            self.optimizer.state_dict(),
        )
        return cap

    def get_uv_mask(self):
        self.uv_mask = load_uv_region_masks("flame_model/assets/flame/uv_region_masks.pkl")
        return self.uv_mask
    
    def uvcoords_sample(self):  # 在 UV 空间中进行采样 为每个绑定点（binding）找到对应的 UV 坐标
        face_index, bary_coords = hybrid_sampling_by_binding(
            binding=self.binding,
            tex_coord=self.flame_model.verts_uvs.to("cuda"),
            uv_faces=self.flame_model.textures_idx.to("cuda"),
            uv_size=256,
        )
        prior_uvcoords_sample = reweight_uvcoords_by_barycoords(
            uvcoords=self.flame_model.verts_uvs.to("cuda"),
            uvfaces=self.flame_model.textures_idx.to("cuda"),
            face_index=face_index,
            bary_coords=bary_coords,
        )
        prior_uvcoords_sample = prior_uvcoords_sample[..., :2]

        return prior_uvcoords_sample

    def get_position_map(self, verts):
        self.map_generator = PositionMapGenerator( # 创建 PositionMapGenerator 对象，用于生成 UV 空间下的位置图（Position Map）
            verts = verts,  # 输入当前 mesh 的顶点坐标
            faces=self.flame_model.faces.unsqueeze(0),   # mesh 三角面索引
            uvfaces=self.flame_model.textures_idx.unsqueeze(0),  # UV 三角面索引
            uv_coords=self.flame_model.verts_uvs.unsqueeze(0),   # UV 顶点坐标
            image_size=256,   # 渲染时使用的图像分辨率
            uv_size=256,  # 输出 UV position map 的分辨率
            device="cuda",   # 指定所有计算在 GPU 上运行
        )
        return self.map_generator.generate_position_map()

    def get_vertex_displace_map(self):
        return self.map_generator.displacement_map(self.verts)
    
def blend_shapes(betas, shape_disps):
    """Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum("bl,mkl->bmk", [betas, shape_disps])
    return blend_shape
    