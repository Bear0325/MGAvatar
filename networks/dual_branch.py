import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch3d.transforms import axis_angle_to_quaternion
from networks.modules import (
    DoubleConv,
    Down,
    Up,
    FourierEncoding3D
)

class DetailBlock(nn.Module):
    """
    A small convolutional block to enhance local high-frequency details.
    """
    def __init__(self, in_ch, expr_dim, hidden_ch=64):
        super().__init__()
        self.expr_proj = nn.Sequential(
            nn.Linear(expr_dim, hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(in_ch + in_ch + hidden_ch, hidden_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(hidden_ch, hidden_ch, 1)

    def forward(self, global_feat, disp_map, expr, mask):
        """
        global_feat: [B, C, H, W]
        disp_map:    [B, C, H, W]
        expr:        [B, expr_dim]
        mask:        [1, 1, H, W]  (binary mask)
        """
        B, C, H, W = global_feat.shape

        e = self.expr_proj(expr).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        disp_map_masked = disp_map * mask
        global_feat_masked = global_feat * mask

        x = torch.cat([disp_map_masked, global_feat_masked, e], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv_out(x)

        return x * mask

# class DualBranchUNet(nn.Module): # 激进魔改版：去参考图分支 + Mask合流自适应网络 -30000:31.0207  24.6146
#     def __init__(
#         self,
#         device, 
#         uv_sample_coords,  # 保持原接口
#         uv_mask,  # 保持原接口
#         reference_image=None,  # 保持原接口，内部不再使用
#         position_map=None,   # 保持原接口
#         condition_dim=118,  
#         bilinear=True, 
#     ):
#         super().__init__()  
#         self.device = device   
#         self.condition_dim = condition_dim  
#         self.bilinear = bilinear

#         # 1. 将 4 个区域的 Mask 融合成一个 4 通道的空间先验图 [1, 4, 256, 256]
#         # 这样网络能一次性感知所有关键区域，不再需要走 4 个独立的局部解码网络
#         eye_m = uv_mask["eye_region"].float().to(device)
#         nose_m = uv_mask["nose"].float().to(device)
#         lips_m = uv_mask["lips"].float().to(device)
#         fore_m = uv_mask["forehead"].float().to(device)
#         self.register_buffer("spatial_mask_prior", torch.cat([eye_m, nose_m, lips_m, fore_m], dim=1))

#         # 保持原有的图像预处理和 buffer 注册接口，防止外部调用报错，但 forward 内部不再依赖它们
#         self.transform = T.Compose([T.Resize((256, 256), antialias=True), T.ToTensor()])   
#         self.resize = T.Resize((256, 256), antialias=True)   
#         if reference_image is not None:  
#             self.register_buffer("reference_image", self.transform(reference_image).unsqueeze(0).detach().to(device))  
#         else:
#             self.register_buffer("reference_image", torch.empty(1, 3, 256, 256)) 
#         if position_map is not None:  
#             self.register_buffer("position_map", position_map.detach().float().to(device))   
#         else:
#             self.register_buffer("position_map", torch.empty(1, 3, 256, 256)) 

#         self.uv_sample_coords = uv_sample_coords.detach().float().to(device)

#         #=========== 2. 砍掉原有的 UNet 编码器，建立轻量几何骨架网络 ===========
#         self.fourier = FourierEncoding3D(num_bands=4, max_freq=10.0)  # 稍微微调频段到 4，过滤高频噪声
#         # 输入通道: 3 (XYZ) + 4 * 6 * 2 (傅里叶变换后) = 51 通道
#         # 建立一个极其轻量的 2D 卷积网络直接处理位置图，代替原本庞大的 UNet
#         self.geo_encoder = nn.Sequential(
#             nn.Conv2d(3 + 4 * 6, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
        
#         # 同样轻量化位移特征提取层
#         self.disp_conv = nn.Sequential(
#             nn.Conv2d(3 + 4 * 6, 64, kernel_size=3, padding=1), 
#             nn.ReLU(inplace=True)
#         )

#         #=========== 3. FLAME 条件调制层 (AdaIN) ===========
#         self.expr_embed = nn.Sequential(nn.Linear(100, 64), nn.ReLU(inplace=True))   
#         self.pose_embed = nn.Sequential(nn.Linear(15, 32), nn.ReLU(inplace=True)) 
#         self.tran_embed = nn.Sequential(nn.Linear(3, 32), nn.ReLU(inplace=True))    
        
#         # 融合后的条件维度为 64 + 32 + 32 = 128 维
#         # 用于预测 128 个几何通道的 Scale 和 Shift
#         self.flame_modulator = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128 * 2) 
#         )

#         #=========== 4. 合流后的全新大解码器 ===========
#         # 输入通道包括：几何特征(128) + 空间Mask先验(4) + 细节图特征(64) = 196 通道
#         # 一个全图网络直接吞入所有特征，边缘自然过渡，指标更稳健
#         self.unified_decoder = nn.Sequential(
#             nn.Conv2d(128 + 4 + 64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 13, kernel_size=1),  # 原汁原味输出 13 通道 2D 属性图
#         )

#         # 用于预计算几何特征的懒加载 Buffer
#         self.register_buffer("static_geo_feat", None, persistent=False)
#         self._initialize_weights()   

#     def forward(self, flame_param, timestep, displacement_map):
#         # ========== 1. Process condition ==========
#         expr = flame_param["expr"][timestep].unsqueeze(0)  
#         neck_pose = flame_param["neck_pose"][timestep].unsqueeze(0)  
#         jaw_pose = flame_param["jaw_pose"][timestep].unsqueeze(0)  
#         eyes_pose = flame_param["eyes_pose"][timestep].unsqueeze(0)  
#         rotation = flame_param["rotation"][timestep].unsqueeze(0)  
#         pose = torch.cat((neck_pose, jaw_pose, eyes_pose, rotation), dim=1)
#         translation = flame_param["translation"][timestep].unsqueeze(0)  

#         # ========== 2. 优化点 1：位置几何特征懒加载 (摆脱重型 UNet 编码器) ==========
#         if self.static_geo_feat is None:
#             with torch.no_grad():
#                 pos_fourier = self.fourier(self.position_map.float())
#                 self.register_buffer("static_geo_feat", self.geo_encoder(pos_fourier).detach(), persistent=False)

#         # ========== 3. FLAME 条件自适应调制 (无需 Expand) ==========
#         expr_emb = self.expr_embed(expr)  
#         pose_emb = self.pose_embed(pose)  
#         trans_emb = self.tran_embed(translation)  
#         flame_cond = torch.cat([expr_emb, pose_emb, trans_emb], dim=1) 
        
#         # 预测通道调制系数 [1, 256, 1, 1]
#         mod_params = self.flame_modulator(flame_cond).unsqueeze(-1).unsqueeze(-1) 
#         scale, shift = torch.chunk(mod_params, 2, dim=1) 
        
#         # 对预计算的 2D 几何骨架特征直接进行风格化调制
#         geo_modulated = self.static_geo_feat * (1 + scale) + shift 

#         # ========== 4. 处理位移细节图 ==========
#         disp_f = self.fourier(displacement_map.float())  
#         disp = self.disp_conv(disp_f)   

#         # ========== 5. 核心合流解耦：Unified Feature Fusion ==========
#         # 摒弃原本 4 次排队串行的 local_decoder 及其繁琐的特征拼接
#         # 直接把：1)调制后的几何、2)多通道空间Mask、3)位移细节 在通道维度融为一体
#         unified_feat = torch.cat(
#             [geo_modulated, self.spatial_mask_prior, disp], dim=1
#         )
        
#         # 统一大解码器一步到位，输出 [1, 13, 256, 256] 属性图
#         offset_out = self.unified_decoder(unified_feat)  

#         # ========== 6. 尾部后处理 (完美对接你原版的 13->14 通道采样扩充函数) ==========
#         offset_final = self._offset_attr_process(offset_out, self.uv_sample_coords)

#         return offset_final
class DualBranchUNet(nn.Module): # 保持你原本的类名
    def __init__(
        self,
        device, # 运行设备（cuda 或 cpu）
        uv_sample_coords,  # UV采样坐标
        uv_mask,  # 不同区域的UV mask
        reference_image=None,  # 保持接口兼容，外部传入也不再使用
        position_map=None,   # 位置图
        condition_dim=118,  # 条件输入维度
        bilinear=True, # 是否使用双线性插值
    ):
        super().__init__()  # 调用父类初始化
        self.device = device   # 保存设备信息
        self.eye_mask = uv_mask["eye_region"]  # 提取眼睛区域mask
        self.nose_mask = uv_mask["nose"]  # 提取鼻子区域mask
        self.lips_mask = uv_mask["lips"]   # 提取嘴唇区域mask
        self.forehead_mask = uv_mask["forehead"]  # 提取额头区域mask

        self.transform = T.Compose([T.Resize((256, 256), antialias=True), T.ToTensor()])  
        self.resize = T.Resize((256, 256), antialias=True)   

        # 🟢 【核心魔改点】彻底丢掉外部参考图输入！
        # 改为声明一个全图共享的、可学习的 3 通道常数基础图（随机初始化为类似中性灰度或标准分布）
        # 它会随网络一起训练，自动学出最完美的静态人脸特征底图
        self.reference_image = nn.Parameter(torch.randn(1, 3, 256, 256).to(device) * 0.01)

        if position_map is not None:  # 如果输入了位置图
            self.register_buffer("position_map", position_map.detach().float().to(device))   # 注册位置图buffer
        else:
            self.register_buffer("position_map", torch.empty(1, 3, 256, 256)) # 创建空的位置图tensor

        self.uv_sample_coords = uv_sample_coords.detach().float().to(device)  # 保存UV采样坐标并放到设备上
        self.condition_dim = condition_dim  # 保存条件维度
        self.bilinear = bilinear   # 保存是否双线性插值
        factor = 2 if bilinear else 1  # 如果使用双线性插值，则通道缩减因子为2

        #=========== Encoder (完全保持原样) ===========
        self.inc = DoubleConv(3, 64, mid_channels=32)    # 输入层：3通道RGB -> 64通道特征
        self.down1 = Down(64, 128)   # 第一次下采样：64 -> 128
        self.down2 = Down(128, 256)   # 第二次下采样：128 -> 256
        self.down3 = Down(256, 512 // factor)   # 第三次下采样：256 -> 512/factor
        self.up1 = Up(512, 256 // factor, bilinear)  # 第一次上采样：512 -> 256/factor
        self.up2 = Up(256, 128 // factor, bilinear)  # 第二次上采样：256 -> 128/factor
        self.up3 = Up(128, 64, bilinear)  # 第三次上采样：128 -> 64

        #=========== Process different features (完全保持原样) ===========
        self.fourier = FourierEncoding3D(num_bands=6, max_freq=10.0)  # 3D Fourier编码器，用于增强高频信息
        self.uv_conv = nn.Sequential(nn.Conv2d(3 + 6 * 6, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))   # 对UV特征做卷积
        self.disp_conv = nn.Sequential(nn.Conv2d(3 + 6 * 6, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))   # 对位移特征做卷积
        self.expr_embed = nn.Sequential(nn.Linear(100, 96), nn.ReLU(inplace=True))   # 表情参数100维 -> 96维
        self.pose_embed = nn.Sequential(nn.Linear(15, 16), nn.ReLU(inplace=True)) # pose参数15维 -> 16维
        self.tran_embed = nn.Sequential(nn.Linear(3, 16), nn.ReLU(inplace=True))     # 平移参数3维 -> 16维

        #=========== Global Decoder (完全保持原样) ============
        self.global_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 全局特征卷积
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 通道降到128
            nn.ReLU(inplace=True),  # 激活函数
        )

        #=========== Local Decoder (完全保持原样) ============
        self.local_decoder = DetailBlock(in_ch=64, expr_dim=100, hidden_ch=64)  # 输入通道数，表情维度，隐藏层通道

        #=========== Fuse Layer (完全保持原样) ===========
        self.final_fuse = nn.Sequential(
            nn.Conv2d(128 + 4 * 64, 128, kernel_size=3, padding=1),   # 融合全局与局部特征
            nn.ReLU(inplace=True),    # 激活函数
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 通道降到64
            nn.ReLU(inplace=True),   # 激活函数
            nn.Conv2d(64, 13, kernel_size=1),    # 输出13通道结果
        )

        self._initialize_weights()   # 初始化网络权重

    def forward(
        self, flame_param, timestep, displacement_map
    ):
        # process condition
        expr = flame_param["expr"][timestep].unsqueeze(0)  # 1*100 
        neck_pose = flame_param["neck_pose"][timestep].unsqueeze(0)  # 1*3
        jaw_pose = flame_param["jaw_pose"][timestep].unsqueeze(0)  # 1*3
        eyes_pose = flame_param["eyes_pose"][timestep].unsqueeze(0)  # 1*6
        rotation = flame_param["rotation"][timestep].unsqueeze(0)  # 1*3
        pose = torch.cat((neck_pose, jaw_pose, eyes_pose, rotation), dim=1)
        translation = flame_param["translation"][timestep].unsqueeze(0)  # 1*3

        #========== Encoder (完全保留，但输入变为了可学习的底图) ==========
        # 🟢 计算开销减小：因为 self.reference_image 直接驻留在 GPU 上，
        # 训练或推理时彻底免去了外部单张图片读取、图像变换等杂务输入。
        x1 = self.inc(self.reference_image)  # 第一层卷积，提取浅层特征
        x2 = self.down1(x1)   # 第一次下采样
        x3 = self.down2(x2)   # 第二次下采样
        x4 = self.down3(x3)   # 第三次下采样，得到深层特征
        d1 = self.up1(x4, x3)  # 第一次上采样，并与x3做skip connection
        d2 = self.up2(d1, x2)   # 第二次上采样，并与x2做skip connection
        d3 = self.up3(d2, x1)  # 第三次上采样，并与x1做skip connection

        #========== Global Decoder (完全保持原样) ==========
        pos_fourier = self.fourier(self.position_map.float())   # 对position map做Fourier编码
        uv_feat = self.uv_conv(pos_fourier)   # 对Fourier编码后的UV特征做卷积
        expr_emb = self.expr_embed(expr)  # 表情参数 embedding：100维 -> 96维
        expr_feat = (expr_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3)))  # 将表情特征扩展到空间维度
        pose_emb = self.pose_embed(pose)    # pose embedding：15维 -> 16维
        pose_feat = (pose_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3)))   # 扩展pose特征到空间维度
        trans_emb = self.tran_embed(translation)   # translation embedding：3维 -> 16维
        trans_feat = (trans_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3)))  # 扩展translation特征到空间维度
        global_feat = torch.cat([d3, uv_feat, expr_feat, pose_feat, trans_feat], dim=1)   # 拼接所有全局特征
        offset_global = self.global_decoder(global_feat)  # 全局decoder预测全局offset特征

        #========== Local Decoder (完全保持原样) ==========
        disp_f = self.fourier(displacement_map.float())   # 对位移图做Fourier编码
        disp = self.disp_conv(disp_f)   # 对位移特征做卷积

        local_eye = self.local_decoder(d3, disp, expr, self.eye_mask) # 预测眼睛区域局部细节
        local_nose = self.local_decoder(d3, disp, expr, self.nose_mask)   # 预测鼻子区域局部细节
        local_lips = self.local_decoder(d3, disp, expr, self.lips_mask)  # 预测嘴唇区域局部细节
        local_forehead = self.local_decoder(d3, disp, expr, self.forehead_mask) # 预测额头区域局部细节

        #========= Fuse Layer (完全保持原样) =========
        unified_feat = torch.concat(  # 将全局与所有局部特征拼接融合
            [offset_global, local_eye, local_nose, local_lips, local_forehead], dim=1
        )
        offset_out = self.final_fuse(unified_feat)   # 最终融合网络输出13通道offset属性
        offset_final = self._offset_attr_process(offset_out, self.uv_sample_coords)    # 根据UV采样坐标提取最终offset属性

        return offset_final   # 返回最终预测结果
    
    def forward_once(self):
        # for testing, only calculate d3 once
        x1 = self.inc(self.reference_image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        self.d3 = self.up3(d2, x1)
        pos_fourier = self.fourier(self.position_map.float())
        self.uv_feat = self.uv_conv(pos_fourier)


    def _decode(self, flame_param, timestep, displacement_map):
        expr = flame_param["expr"][timestep].unsqueeze(0)  # 1*100
        neck_pose = flame_param["neck_pose"][timestep].unsqueeze(0)  # 1*3
        jaw_pose = flame_param["jaw_pose"][timestep].unsqueeze(0)  # 1*3
        eyes_pose = flame_param["eyes_pose"][timestep].unsqueeze(0)  # 1*6
        rotation = flame_param["rotation"][timestep].unsqueeze(0)  # 1*3
        pose = torch.cat((neck_pose, jaw_pose, eyes_pose, rotation), dim=1)
        translation = flame_param["translation"][timestep].unsqueeze(0)  # 1*3

        #========= Global Decoder ===========
        H, W = self.d3.shape[2], self.d3.shape[3]
        pos_fourier = self.fourier(self.position_map.float())
        uv_feat = self.uv_conv(pos_fourier)
        expr_emb = self.expr_embed(expr)
        expr_feat = (
            expr_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )
        pose_emb = self.pose_embed(pose)
        pose_feat = (
            pose_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )
        trans_emb = self.tran_embed(translation)
        trans_feat = (
            trans_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )
        global_feat = torch.cat([self.d3, uv_feat, expr_feat, pose_feat, trans_feat], dim=1)
        offset_global = self.global_decoder(global_feat)

        #========= Local Decoder ==========
        disp_f = self.fourier(displacement_map.float())
        disp = self.disp_conv(disp_f)

        local_eye = self.local_decoder(self.d3, disp, expr, self.eye_mask)
        local_nose = self.local_decoder(self.d3, disp, expr, self.nose_mask)
        local_lips = self.local_decoder(self.d3, disp, expr, self.lips_mask)
        local_forehead = self.local_decoder(self.d3, disp, expr, self.forehead_mask)

        #========= Fuse Layer ==========
        unified_feat = torch.concat(
            [offset_global, local_eye, local_nose, local_lips, local_forehead], dim=1
        )
        offset_out = self.final_fuse(unified_feat)
        offset_final = self._offset_attr_process(offset_out, self.uv_sample_coords)

        return offset_final

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def update_uv_coords(self, uv_sample_coords):
        self.uv_sample_coords = uv_sample_coords.detach().float().to(self.device)

    def _offset_attr_process(self, offset_map, uv_coords):
        """
        Args:
            offset_map: [B, 13, 512, 512]
            uv_coords:  [B, N, 2]
        Returns:
            offset:     [B, N, 13]
        """
        # sample per-point offset from the map
        offset = self._uv_look_up(texture=offset_map, uv_coords=uv_coords)

        # apply activations to individual channels
        return torch.cat(
            [
                self._offset_position_activation(offset[:, :, 0:3]),
                self._offset_scaling_activation(offset[:, :, 3:6]),
                self._rotation_activation(offset[:, :, 6:9]),  # (B, N, 4)
                self._offset_color_activation(offset[:, :, 9:12]),
                self._offset_opacity_activation(offset[:, :, 12:13]),
            ],
            dim=-1,
        )

    @staticmethod
    def _offset_position_activation(input):
        """
        Force the NN output serves as position to be between [-0.1, 0.1]
        """
        return torch.tanh(input) * 0.1

    @staticmethod
    def _offset_color_activation(input):
        """
        Force the NN output serves as color to be between [-0.7, 0.7].
        """
        return torch.tanh(input) * 0.7

    @staticmethod
    def _offset_scaling_activation(input):
        """
        Force the NN output serves as scaling within proper range [0, +inf].
        """
        return torch.exp(input)

    @staticmethod
    def _offset_opacity_activation(input):
        """
        Force the NN output serves as scaling within proper range [-0.5, 0.5].
        """
        return torch.tanh(input) * 0.5

    def _rotation_activation(self, input: torch.Tensor) -> torch.Tensor:
        """
        Convert NN output to quaternion rotation [r, x, y, z] with proper normalization.
        Supports only [B, N, 3]input.

        Args:
            input: Tensor of shape [B, N, 3] — axis-angle representation.
        Returns:
            Tensor of shape [B, N, 4] — quaternion in [r, x, y, z] format.
        """
        input = torch.tanh(input) * (torch.pi)  # scale to [-π, π]
        quat = axis_angle_to_quaternion(input)  # [B, N, 4] wxyz

        return quat.contiguous()

    def _uv_look_up(
        self, texture: torch.Tensor, uv_coords: torch.Tensor
    ) -> torch.Tensor:
        """Use sampling to get gs value
        Args:
            texture:    [B, C, H, W] texture color
            uv_coords:  [B, N, 2] texture uv
        Returns:
            values:     [B, N, C] texture sampling
        """
        grid = uv_coords * 2.0 - 1.0  # [B, N, 2]
        grid = grid.unsqueeze(2)  # [B, N, 1, 2]

        sampled = F.grid_sample(
            texture, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        sampled_value = sampled.squeeze(3).permute(0, 2, 1)  # [B, N, C]

        return sampled_value