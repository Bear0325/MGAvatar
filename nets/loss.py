import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import lpips
from pytorch3d.structures import Meshes
import cv2
import sys
import os
from random import randint 

# class DepthLoss(nn.Module):
#     def __init__(self):
#         super(DepthLoss, self).__init__()
#     def pearson_corrcoef(self, x, y): # 计算皮尔逊相关系数 (Pearson correlation coefficient)
#         x = x - x.mean() # 输入: x, y 为 1D 向量 (已展平的预测与GT)
#         y = y - y.mean()
#         return torch.sum(x * y) / (torch.norm(x) * torch.norm(y) + 1e-8)  # 输出: [-1, 1] 之间的相关系数，越接近1相关性越高
#     # def normalize(self, input, mean=None, std=None): # 对输入做归一化
#     #     input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
#     #     input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
#     #     return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))
#     def normalize(self, input, mean=None, std=None):
#         input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
#         input_std = torch.std(input, dim=1, keepdim=True) if std is None else std

#         # 防止除以 0：对过小的 std 加一个最小阈值
#         eps = 1e-6
#         input_std = torch.clamp(input_std, min=eps)

#         return (input - input_mean) / input_std

#     def patchify(self, input, patch_size):  # 将输入划分为 patch
#         patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size) # [B, C*P*P, L]→[B, L, C*P*P]→[B*L, P*P]
#         return patches
    
#     def patch_norm_mse_loss(self, input, target, patch_size, margin, return_mask=False): # Patch 级别的归一化 + L2损失 (局部)
#         input_patches = self.normalize(self.patchify(input, patch_size))  # 对每个patch独立归一化
#         target_patches = self.normalize(self.patchify(target, patch_size))
#         return self.margin_l2_loss(input_patches, target_patches, margin, return_mask)
#         # return pearson_correlation_loss(input_patches, target_patches, margin, return_mask)

#     def patch_norm_mse_loss_global(self, input, target, patch_size, margin, return_mask=False): # Patch 级别的归一化 + L2损失 (全局)
#         input_patches = self.normalize(self.patchify(input, patch_size), std = input.std().detach()) # 用整张图的std来归一化，而不是每个patch单独的std
#         target_patches = self.normalize(self.patchify(target, patch_size), std = target.std().detach())
#         return self.margin_l2_loss(input_patches, target_patches, margin, return_mask)
#         # return pearson_correlation_loss(input_patches, target_patches, margin, return_mask)

#     def margin_l2_loss(self, depth_out, depth_tgt, margin, return_mask=False):  # 带有 margin 的 L2 损失
#         mask = (depth_out - depth_tgt).abs() > margin # 仅在 (预测 - GT) > margin 的地方计算平方误差
#         if not return_mask:
#             return ((depth_out - depth_tgt)[mask] ** 2).mean()
#         else:
#             return ((depth_out - depth_tgt)[mask] ** 2).mean(), mask

#     def forward(self, depth_out, depth_tgt, patch_size, margin=0.02):
#         if depth_out.dim() == 3:
#             depth_out = depth_out.unsqueeze(1) 
#         if depth_tgt.dim() == 3:
#             depth_tgt = depth_tgt.unsqueeze(1)
#         local_loss = self.patch_norm_mse_loss(depth_out, depth_tgt, patch_size, margin)
#         global_loss = self.patch_norm_mse_loss_global(depth_out, depth_tgt, patch_size, margin)
        
#         # 调用皮尔逊相关系数
#         pred_flat = depth_out.view(-1)
#         gt_flat = depth_tgt.view(-1)
#         pearson_loss = 1.0 - self.pearson_corrcoef(pred_flat, gt_flat)
        
#         total_loss = cfg.depth_local_loss_weight * local_loss + cfg.depth_global_loss_weight * global_loss + cfg.depth_pearson_loss_weight * pearson_loss
#         return total_loss
#原始loss
# class NormalLoss(nn.Module):
#     def __init__(self):
#         super(NormalLoss, self).__init__()

#     def forward(self, normal_out, normal_tgt):
#         if normal_tgt.dim() == 3:
#             normal_tgt = normal_tgt.unsqueeze(0)
#         is_valid_out = 1 - ((normal_out==0).sum(1)[:,None] == 3).float()
#         is_valid_tgt = 1 - ((normal_tgt==0).sum(1)[:,None] == 3).float()
#         normal_out = normal_out*2-1 # [0,1] -> [-1,1]
#         normal_tgt = normal_tgt*2-1 # [0,1] -> [-1,1]
#         # loss = (torch.abs(normal_out - normal_tgt) + (1 - torch.sum(normal_out*normal_tgt,1)[:,None])) * is_valid_out * is_valid_tgt
#         loss = torch.abs(normal_out - normal_tgt)
#         return loss

#给原来的loss添加归一化 没有用 还是飘
# class NormalLoss(nn.Module):
#     def __init__(self):
#         super(NormalLoss, self).__init__()

#     def forward(self, normal_out, normal_tgt):
#         if normal_tgt.dim() == 3:
#             normal_tgt = normal_tgt.unsqueeze(0)
#         is_valid_out = 1 - ((normal_out==0).sum(1)[:,None] == 3).float()
#         is_valid_tgt = 1 - ((normal_tgt==0).sum(1)[:,None] == 3).float()
#         normal_out = normal_out*2-1 # [0,1] -> [-1,1]
#         normal_tgt = normal_tgt*2-1 # [0,1] -> [-1,1]
#         # 单位化，确保方向对齐
#         normal_out = F.normalize(normal_out, dim=1, eps=1e-8)
#         normal_tgt = F.normalize(normal_tgt, dim=1, eps=1e-8)
#         loss = (torch.abs(normal_out - normal_tgt) + (1 - torch.sum(normal_out*normal_tgt,1)[:,None])) * is_valid_out * is_valid_tgt
#         # loss = torch.abs(normal_out - normal_tgt)
#         return loss
    
# #chatgpt改良版loss 最新版本loss
# class NormalLoss(nn.Module):
#     def __init__(self, angle_weight=0.5, eps=1e-6):
#         """
#         参数:
#             angle_weight: 角度损失权重 (建议 0.3~0.7)
#             eps: 避免除0错误的小常数
#         """
#         super(NormalLoss, self).__init__()
#         self.angle_weight = angle_weight
#         self.eps = eps

#     def forward(self, normal_out, normal_tgt):
#         """
#         输入:
#             normal_out: 模型渲染出的法线图 (B,3,H,W) 或 (3,H,W)
#             normal_tgt: 目标法线图 (B,3,H,W) 或 (3,H,W)
#         返回:
#             loss_map: 每像素法线损失 (未平均)
#         """
#         if normal_tgt.dim() == 3:
#             normal_tgt = normal_tgt.unsqueeze(0)
#         if normal_out.dim() == 3:
#             normal_out = normal_out.unsqueeze(0)

#         # [0,1] → [-1,1]
#         normal_out = normal_out * 2 - 1
#         normal_tgt = normal_tgt * 2 - 1

#         # 单位化，确保方向对齐
#         normal_out = F.normalize(normal_out, dim=1, eps=self.eps)
#         normal_tgt = F.normalize(normal_tgt, dim=1, eps=self.eps)

#         # 有效 mask（排除背景或空像素）
#         mask_out = (normal_out.abs().sum(1, keepdim=True) > 1e-4).float()
#         mask_tgt = (normal_tgt.abs().sum(1, keepdim=True) > 1e-4).float()
#         mask = (mask_out * mask_tgt).detach()

#         # L1 差异项
#         l1_loss = torch.abs(normal_out - normal_tgt)

#         # 角度差项（cosine similarity）
#         cos_sim = torch.sum(normal_out * normal_tgt, dim=1, keepdim=True).clamp(-1.0, 1.0)
#         angle_loss = 1 - cos_sim

#         # 综合损失
#         loss_map = (l1_loss + self.angle_weight * angle_loss) * mask
#         return loss_map
    
# class GeoLoss(nn.Module):
#     def __init__(self):
#         super(GeoLoss, self).__init__()
#         self.depth_loss = DepthLoss()
#         self.normal_loss = NormalLoss()

#     def forward(self, depth_out, normal_out, depth_tgt, normal_tgt):
#         #patch-depth
#         patch_range = (5, 17)
#         depth_loss = self.depth_loss(depth_out, depth_tgt, randint(patch_range[0], patch_range[1]), 0.02) * cfg.depth_loss_weight

#         # depth_loss = self.depth_loss(depth_out, depth_tgt) * cfg.depth_loss_weight
#         normal_loss = self.normal_loss(normal_out, normal_tgt) * cfg.normal_loss_weight
#         loss = depth_loss + normal_loss 
#         # loss = depth_loss
#         # loss = normal_loss
#         return loss

# #只有法线约束   
# class GeoLoss(nn.Module):
#     def __init__(self):
#         super(GeoLoss, self).__init__()
#         # self.depth_loss = DepthLoss()
#         self.normal_loss = NormalLoss()

#     def forward(self, normal_out, normal_tgt):
#         #patch-depth
#         # patch_range = (5, 17)
#         # depth_loss = self.depth_loss(depth_out, depth_tgt, randint(patch_range[0], patch_range[1]), 0.02) * cfg.depth_loss_weight

#         # depth_loss = self.depth_loss(depth_out, depth_tgt) * cfg.depth_loss_weight
#         normal_loss = self.normal_loss(normal_out, normal_tgt) * cfg.normal_loss_weight
#         # loss = depth_loss + normal_loss 
#         # loss = depth_loss
#         # loss = normal_loss
#         return normal_loss
    
           
class RGBLoss(nn.Module):
    def __init__(self):
        super(RGBLoss, self).__init__()
    
    def forward(self, img_out, img_target):
        loss = torch.abs(img_out - img_target)
        return loss

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.FloatTensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).cuda()
        return gauss / gauss.sum()

    def create_window(self, window_size, feat_dim):
        window_1d = self.gaussian(window_size, 1.5)[:,None]
        window_2d = torch.mm(window_1d, window_1d.permute(1,0))[None,None,:,:]
        window_2d = window_2d.repeat(feat_dim,1,1,1)
        return window_2d

    def forward(self, img_out, img_target, window_size=11):
        _, feat_dim, _, _, = img_out.shape
        window = self.create_window(window_size, feat_dim)
        mu1 = F.conv2d(img_out, window, padding=window_size//2, groups=feat_dim)
        mu2 = F.conv2d(img_target, window, padding=window_size//2, groups=feat_dim)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img_out*img_out, window, padding=window_size//2, groups=feat_dim) - mu1_sq
        sigma2_sq = F.conv2d(img_target*img_target, window, padding=window_size//2, groups=feat_dim) - mu2_sq
        sigma1_sigma2 = F.conv2d(img_out*img_target, window, padding=window_size//2, groups=feat_dim) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1_sigma2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map

# image perceptual loss (LPIPS. https://github.com/richzhang/PerceptualSimilarity)
class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.lpips = lpips.LPIPS(net='vgg').cuda()
        # self.lpips = lpips.LPIPS(net='alex').cuda()

    def forward(self, img_out, img_target):
        img_out = img_out * 2 - 1 # [0,1] -> [-1,1]
        img_target = img_target * 2 - 1 # [0,1] -> [-1,1]
        loss = self.lpips(img_out, img_target)
        return loss

class LaplacianReg(nn.Module):
    def __init__(self, vertex_num, face):
        super(LaplacianReg, self).__init__()
        self.neighbor_idxs, self.neighbor_weights = self.get_neighbor(vertex_num, face)

    def get_neighbor(self, vertex_num, face, neighbor_max_num = 10):
        adj = {i: set() for i in range(vertex_num)}
        for i in range(len(face)):
            for idx in face[i]:
                adj[idx] |= set(face[i]) - set([idx])

        neighbor_idxs = np.tile(np.arange(vertex_num)[:,None], (1, neighbor_max_num))
        neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)
        for idx in range(vertex_num):
            neighbor_num = min(len(adj[idx]), neighbor_max_num)
            neighbor_idxs[idx,:neighbor_num] = np.array(list(adj[idx]))[:neighbor_num]
            neighbor_weights[idx,:neighbor_num] = -1.0 / neighbor_num
        
        neighbor_idxs, neighbor_weights = torch.from_numpy(neighbor_idxs).cuda(), torch.from_numpy(neighbor_weights).cuda()
        return neighbor_idxs, neighbor_weights
    
    def compute_laplacian(self, x, neighbor_idxs, neighbor_weights):
        # lap = x + (x[neighbor_idxs] * neighbor_weights[:, :, None]).sum(2)
        lap = x + (x[neighbor_idxs] * neighbor_weights[:, :, None]).sum(1)
        return lap

    def forward(self, out, target):
        if target is None:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            loss = lap_out ** 2
            return loss
        else:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            lap_target = self.compute_laplacian(target, self.neighbor_idxs, self.neighbor_weights)
            loss = (lap_out - lap_target) ** 2
            return loss
