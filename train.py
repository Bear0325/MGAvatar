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

import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_normal
# from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from pathlib import Path
import torchvision

import numpy as np 

from networks.dual_branch import DualBranchUNet
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)
        # mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # # #########################################

    # labels = np.load(dataset.source_path + '/train_cluster_16.npz')['labels'] # 加载聚类标签数据
    # unique_labels, counts = np.unique(labels, return_counts=True)  # 计算每个唯一标签的出现次数
    # weights = 1.0 / counts # 计算每个标签的采样权重（出现次数越少，权重越高）
    # weights_dict = dict(zip(unique_labels, weights)) # 创建标签到权重的映射字典
    
    # prob_lst = []  # 为每个训练相机计算采样概率
    # train_cameras = scene.getTrainCameras().cameras
    # for cam in train_cameras:
    #     timestep = cam.timestep
    #     prob_lst.append(weights_dict[labels[timestep]])
    # prob_lst = torch.tensor(prob_lst).double() # 将概率列表转换为张量并归一化
    # prob_lst /= torch.sum(prob_lst)


    # sampler = WeightedRandomSampler(weights = prob_lst, num_samples = len(train_cameras), replacement = True) # 创建加权随机采样器
    # loader_camera_train = DataLoader(scene.getTrainCameras(), sampler = sampler, batch_size=None, num_workers=8, pin_memory=True, persistent_workers=True)  # 创建数据加载器，使用加权采样

    # # #########################################
    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    # viewpoint_stack = None
    ema_loss_for_log = 0.0

    print("==================================")
    print("===== Stage 1 Training Start =====")
    print("==================================")
    progress_bar = tqdm(range(first_iter, opt.iteration_stage1), desc="Training progress")
    first_iter += 1
    # ✅ 初始化 TensorBoard writer
    writer = SummaryWriter(log_dir=args.model_path + '/logs')

    for iteration in range(first_iter, opt.iteration_stage1 + 1):        

        iter_start.record()  # 8. 记录迭代开始时间（CUDA事件）

        gaussians.update_learning_rate(iteration, opt) # 9. 更新学习率

        try: # 11. 获取下一个训练相机（viewpoint）
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        assets, assets_refined, offsets = gaussians.forward(viewpoint_cam.timestep)
        
        # head_assets, head_assets_refined, head_offsets = {}, {}, {}
        # key_list = ['mean_3d', 'rotation']   # 需要处理的关键字段

        # # gather assets
        # for key in key_list:
        #     if key not in head_assets:
        #         head_assets[key] = [assets[key]]
        #         head_assets_refined[key] = [assets_refined[key]]
        #     else:
        #         head_assets[key].append(assets[key])
        #         head_assets_refined[key].append(assets_refined[key])
        # # 4. 收集偏移量资产
        # # gather offsets
        # for key in ['mean_offset', 'mean_offset_offset']:
        #     if key not in head_offsets:
        #         head_offsets[key] = [offsets[key]]
        #     else:
        #         head_offsets[key].append(offsets[key])

        # Render # 渲染图像
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        head_render = render(viewpoint_cam, gaussians, pipe, background, assets)# 执行渲染，获取渲染结果包
       
        head_render_refined = render(viewpoint_cam, gaussians, pipe, background, assets_refined)  
        
        # head_render_refined['normalmap'] = render_normal(gaussians.flame_model, viewpoint_cam, gaussians, pipe, background, assets_refined)
        # head_render_refined['normalmap'] = render_normal(gaussians.flame_model, assets_refined, viewpoint_cam, pipe)
        # #调试
        # if iteration in [100, 200, 500, 800, 1000, 2000, 3000, 6000, 8000, 10000, 12000, 15000, 18000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000]:
        #     torchvision.utils.save_image(head_render["render"].detach().cpu(), f"debug_output/debug_output1/stage1/render_iter{iteration}.png")
            # torchvision.utils.save_image(head_render_refined["render"].detach().cpu(), f"debug_output/debug_output1/stage1/render_refined_iter{iteration}.png")
        # #     # save_color_depthmap_transparent(head_render_refined["depthmap"].detach().cpu().numpy(), f"debug_output/debug_output15/depthmap_refined_iter{iteration}.png")
        # #     # torchvision.utils.save_image(head_render_refined["depthmap"].detach().cpu(), f"debug_output/debug_output27/depthmap_refined_iter{iteration}.png")
        #     torchvision.utils.save_image(head_render_refined["normalmap"].detach().cpu(), f"debug_output/debug_output1/stage1/normalmap_refined_iter{iteration}.png")
        # #     # torchvision.utils.save_image(torch.from_numpy(viewpoint_cam.normalmap).float().detach().cpu(), f"debug_output/debug_output1/stage1/normalmap_gt{iteration}.png")
        

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        losses = {}
        losses['l1'] = l1_loss(head_render["render"], gt_image) * (1.0 - opt.lambda_dssim) # 3. L1 损失
        losses['ssim'] = (1.0 - ssim(head_render["render"], gt_image)) * opt.lambda_dssim # 4. DSSIM / SSIM 损失

        losses['l1_refined'] = l1_loss(head_render_refined["render"], gt_image) * (1.0 - opt.lambda_dssim) # 3. L1 损失
        losses['ssim_refined'] = (1.0 - ssim(head_render_refined["render"], gt_image)) * opt.lambda_dssim # 4. DSSIM / SSIM 损失
        # losses['rgb'] = gaussians.rgb_loss(head_render["render"], gt_image) * opt.rgb_loss_weight
        # losses['ssim'] = (1.0 - gaussians.ssim(head_render["render"], gt_image)) * opt.ssim_loss_weight
        # losses['lpips'] = gaussians.lpips(head_renders["render"], gt_image) * cfg.lpips_loss_weight

        # losses['rgb_refined'] = gaussians.rgb_loss(head_render_refined["render"], gt_image) * opt.rgb_loss_weight
        # losses['ssim_refined'] = (1.0 - gaussians.ssim(head_render_refined["render"], gt_image)) * opt.ssim_loss_weight
        # losses['lpips_refined'] = gaussians.lpips(head_renders_refined["render"], gt_image) * cfg.lpips_loss_weight
        
        # 设置不同身体部位的权重（手部权重高，面部权重中等）
        weight = torch.ones((gaussians.flame_model.vertex_num_upsampled,1)).float().cuda() * 10
        weight[gaussians.is_face_expr,:] = 10 #10 # 表情区域中等权重
        # 均值偏移正则化
        losses['gaussian_mean_reg'] = (offsets['mean_offset'] ** 2 + offsets['mean_offset_offset'] ** 2) * weight
        #尺度正则化权重设置
        weight = torch.ones((gaussians.flame_model.vertex_num_upsampled,1)).float().cuda()
        weight[gaussians.is_face_expr,:] = 10 #10
     
        # 尺度正则化（不区分预热阶段和正常阶段）
        losses['gaussian_scale_reg'] = (gaussians.get_scaling_mesh ** 2) * weight


        # 12. 拉普拉斯平滑正则化（保持网格平滑）
        # 均值拉普拉斯正则化
        weight = torch.ones((gaussians.flame_model.vertex_num_upsampled,1)).float().cuda()
        weight[gaussians.is_face_expr,:] = 50 # 表情区域高权重
        losses['lap_mean'] = (gaussians.lap_reg(gaussians.xyz_cano + offsets['mean_offset'], gaussians.xyz_cano) + \
                                gaussians.lap_reg(gaussians.xyz_cano + offsets['mean_offset'] + offsets['mean_offset_offset'], gaussians.xyz_cano)) * 100000 * weight
        # 尺度拉普拉斯正则化
        weight = torch.ones((gaussians.flame_model.vertex_num_upsampled,1)).float().cuda()
        losses['lap_scale'] = (gaussians.lap_reg(gaussians.get_scaling_mesh, None) + gaussians.lap_reg(gaussians.get_scaling_mesh, None)) * 100000 * weight



#======================================================================================================================================
        # # 12. 调整权重
        # # 均值拉普拉斯正则化
        # weight = torch.ones((gaussians.flame_model.vertex_num_upsampled,1)).float().cuda()
        # weight[gaussians.is_face_expr,:] = 50 # 表情区域高权重
        # losses['lap_mean'] = (gaussians.lap_reg(gaussians.xyz_cano + offsets['mean_offset'], gaussians.xyz_cano) + \
        #                         gaussians.lap_reg(gaussians.xyz_cano + offsets['mean_offset'] + offsets['mean_offset_offset'], gaussians.xyz_cano)) * 10000 * weight
        # # 尺度拉普拉斯正则化
        # weight = torch.ones((gaussians.flame_model.vertex_num_upsampled,1)).float().cuda()
        # losses['lap_scale'] = (gaussians.lap_reg(gaussians.get_scaling_mesh, None) + gaussians.lap_reg(gaussians.get_scaling_mesh, None)) * 10000 * weight


        # # 12. 拉普拉斯平滑正则化（保持网格平滑） 第二个地方传none
        # # 均值拉普拉斯正则化
        # weight = torch.ones((gaussians.flame_model.vertex_num_upsampled,1)).float().cuda()
        # weight[gaussians.is_face_expr,:] = 50 # 表情区域高权重
        # losses['lap_mean'] = (gaussians.lap_reg(gaussians.xyz_cano + offsets['mean_offset'], None) + \
        #                         gaussians.lap_reg(gaussians.xyz_cano + offsets['mean_offset'] + offsets['mean_offset_offset'], None)) * 1000 * weight
        # # 尺度拉普拉斯正则化
        # weight = torch.ones((gaussians.flame_model.vertex_num_upsampled,1)).float().cuda()
        # losses['lap_scale'] = (gaussians.lap_reg(gaussians.get_scaling_mesh, None) + gaussians.lap_reg(gaussians.get_scaling_mesh, None)) * 100000 * weight


        

        losses = {k: losses[k].mean() for k in losses}
        losses['total'] = sum(losses[k] for k in losses)
        losses['total'].backward()


        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iteration_stage1:
                progress_bar.close()

            testing_iterations = [100000]
            # testing_iterations = [2]
            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # Optimizer step
            if iteration < opt.iteration_stage1:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # ✅ 把 loss 写入 TensorBoard
            for k, v in losses.items():
                writer.add_scalar(f'Loss_stage1/{k}', v.item(), iteration)
            writer.add_scalar('Loss_stage1/total', losses['total'].item(), iteration)

            # if iteration in {1000}:
            if iteration in {100000}:
            # if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                save_mesh(gaussians.template,gaussians.flame_model.face_upsampled, scene.model_path + "/template.obj")
                save_mesh(gaussians.xyz_cano + offsets['mean_offset'], gaussians.flame_model.face_upsampled, scene.model_path + "/1.obj")
                save_mesh(gaussians.xyz_cano + offsets['mean_offset']+ offsets['mean_offset_offset'], gaussians.flame_model.face_upsampled, scene.model_path + "/2.obj")

    print("==================================")
    print("===== Stage 2 Training Start =====")
    print("==================================")
    
    second_iter = 0
    progress_bar = tqdm(range(second_iter, opt.iteration_stage2), desc="Training progress") # 创建进度条
    second_iter += 1

    # # # #调试用
    # # ================= Load Stage 1 Checkpoint =================
    # ckpt_path = scene.model_path + "/chkpnt100000.pth"   # 或 80000

    # print(f"[Stage2] Loading Stage1 checkpoint from {ckpt_path}")
    # ((gaussian_sd, optimizer_sd), loaded_iter) = torch.load(
    #     ckpt_path, map_location="cuda"
    # )
    # # 1. 恢复高斯参数（必须）
    # gaussians.load_state_dict(gaussian_sd, strict=False)
    # gaussians.xyz_cano = gaussian_sd["xyz_cano"].clone() 

    gaussians.init_stage2()
    gaussians.training_setup_stage2(opt)
    gaussians.stage2_is_on = True
    gaussians.stage_offset = False
    
    for iteration in range(second_iter, opt.iteration_stage2 + 1):  
        
        if iteration >= 0 and gaussians.stage_offset == False:
            cano_position = gaussians.xyz_cano[:gaussians.flame_model.vertex_num, :].clone().unsqueeze(0)
            position_map = gaussians.get_position_map(cano_position)
            uv_mask = gaussians.get_uv_mask()
            uv_coords = gaussians.uvcoords_sample()
            # ref_image_path = scene.test_cameras[1.0][8].image_path
            # reference_image = Image.open(ref_image_path).convert("RGB")
            #========== Initialize the network (DualBranchUNet) ==========
            gaussians.dual_branch_net = DualBranchUNet(
                device="cuda",
                uv_sample_coords=uv_coords,
                uv_mask=uv_mask,
                reference_image=None,
                position_map=position_map,
            )
            gaussians.dual_branch_net.to("cuda")
            gaussians.dual_branch_net.train()
            gaussians.optimizer_dual_branch_net = torch.optim.Adam(gaussians.dual_branch_net.parameters(), lr=opt.dual_branch_lr)
            gaussians.stage_offset = True

        iter_start.record()  # 记录迭代开始时间

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree() # 10. 每 1000 次迭代增加 SH 分辨率（up SH degree）

        try: # 11. 获取下一个训练相机（viewpoint）
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        
        gaussians.update_learning_rate(iteration, opt)
        if gaussians.binding != None: #当存在绑定关系的时候，每一帧都要选一个网格
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

            if gaussians.stage_offset == True:
                displacement_map = gaussians.get_vertex_displace_map()
                offset = gaussians.dual_branch_net(gaussians.flame_param, viewpoint_cam.timestep, displacement_map)
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, offset=offset)
            else:
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
    
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"] 

        # Loss
        gt_image = viewpoint_cam.original_image.cuda() # 1. 获取当前相机的 ground truth 图像

        losses = {} # 2. 初始化损失字典
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim) # 3. L1 损失
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim # 4. DSSIM / SSIM 损失

        if gaussians.stage_offset == True:
            losses["offset_scale_reg"] = (torch.abs(offset[0, :, 3:6] - 1).mean() * opt.lambda_offset_scale_reg)
            losses["offset_color_reg"] = (torch.abs(offset[0, :, 10:13]).mean() * opt.lambda_offset_color_reg)
        
        # 5. 高斯点 / mesh 约束
        if gaussians.binding != None: 
            # ----- 5.1 位置约束 (xyz) -----
            if opt.metric_xyz:  # 如果 metric_xyz=True，用尺度归一化后的 xyz 与阈值比较
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:  # 否则直接计算 xyz 范数超过阈值部分的平均值
                # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz  #执行
            # ----- 5.2 尺度约束 (scale) -----
            if opt.lambda_scale != 0:
                if opt.metric_scale: # 使用 get_scaling 方法计算尺度并与阈值比较
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else: # 使用 _scaling 参数（log-space 或 exp）计算尺度
                    # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale #执行
            # ----- 5.3 动态偏移约束 -----
            if opt.lambda_dynamic_offset != 0:  # 计算高斯点随时间的动态偏移损失
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep  # 计算当前帧及前后帧的动态偏移标准差
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std  # 使用所有帧的 dynamic_offset 标准差作为正则化
            # ----- 5.4 Laplacian 平滑约束 -----
            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian  # 保持高斯点分布的平滑性
        
        losses['total'] = sum([v for k, v in losses.items()])
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iteration_stage2:
                progress_bar.close()

            # testing_iterations = [2,3]
            testing_iterations = [60000, 80000, 120000, 150000, 160000, 180000, 190000, 200000, 240000, 300000, 360000, 420000, 480000, 540000, 600000]
            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
   

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.gaussians.cameras_extent, size_threshold)

                    if gaussians.stage_offset == True:
                        uv_coords = gaussians.uvcoords_sample()
                        gaussians.dual_branch_net.update_uv_coords(uv_coords) 
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iteration_stage2:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer_net.step()
                gaussians.optimizer_net.zero_grad(set_to_none = True)
                gaussians.scheduler_net.step()
                if gaussians.stage_offset == True:
                    gaussians.optimizer_dual_branch_net.step()
                    gaussians.optimizer_dual_branch_net.zero_grad(set_to_none=True)

            # ✅ 把 loss 写入 TensorBoard
            for k, v in losses.items():
                writer.add_scalar(f'Loss_stage2/{k}', v.item(), iteration)
            writer.add_scalar('Loss_stage2/total', losses['total'].item(), iteration)

            if iteration in {240000, 300000, 360000}:
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save(uv_coords, scene.model_path + "/uv_coords" + str(iteration)+ ".pt")

import numpy as np
import torch
import trimesh 
def save_mesh(vertices, faces, filename="output.obj"):
    """
    保存mesh为.obj文件

    参数:
        vertices: 顶点坐标, shape = (N, 3)，支持 torch.Tensor 或 numpy.ndarray
        faces:    三角面片索引, shape = (F, 3)，支持 torch.Tensor 或 numpy.ndarray
        filename: 保存的文件路径 (默认: output.obj)
    """
    # 转 numpy
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    # 构建网格
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # 保存
    mesh.export(filename)
    # print(f"✅ Mesh 已保存到: {filename}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    # if tb_writer:
    #     tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
    #     tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
    #     if 'xyz' in losses:
    #         tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
    #     if 'scale' in losses:
    #         tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
    #     if 'dynamic_offset' in losses:
    #         tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
    #     if 'laplacian' in losses:
    #         tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
    #     if 'dynamic_offset_std' in losses:
    #         tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
    #     tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
    #     tb_writer.add_scalar('iter_time', elapsed, iteration)

    log_file = Path(args.model_path) / f"eval_log.txt"
    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            render_path = Path(args.model_path) / config['name'] / "renders"
            mesh_path = Path(args.model_path) / config['name'] / "mesh"
            normal_path = Path(args.model_path) / config['name'] / "normal"
            gts_path = Path(args.model_path) / config['name'] / "gt"
            render_path.mkdir(parents=True, exist_ok=True)
            mesh_path.mkdir(parents=True, exist_ok=True)
            normal_path.mkdir(parents=True, exist_ok=True)
            gts_path.mkdir(parents=True, exist_ok=True)

            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        if scene.gaussians.stage2_is_on:
                            scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)

                            if scene.gaussians.stage_offset == True:
                                displacement_map = scene.gaussians.get_vertex_displace_map()
                                offset = scene.gaussians.dual_branch_net(scene.gaussians.flame_param, viewpoint.timestep, displacement_map)
                                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, offset=offset)["render"], 0.0, 1.0)
                            else:
                                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                            
                        else:
                            _, assets_refined, _ = scene.gaussians.forward(viewpoint.timestep)
                            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, assets_refined)["render"], 0.0, 1.0) # 渲染当前视角
                            # 保存法线和mesh的代码:
                            # normal = torch.clamp(render_normal(scene.gaussians.flame_model, viewpoint, scene.gaussians, *renderArgs, assets_refined), 0.0, 1.0)
                            # save_mesh(assets_refined['mean_3d'], scene.gaussians.flame_model.face_upsampled, mesh_path / f'{idx:05d}.obj')
                            # torchvision.utils.save_image(normal.detach().cpu(), normal_path / f'{idx:05d}.png')
                    
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # # #保存代码 占用空间 先不保存
                    # torchvision.utils.save_image(image.detach().cpu(), render_path / f'{idx:05d}.png')
                    # torchvision.utils.save_image(gt_image.detach().cpu(), gts_path / f'{idx:05d}.png')
                    

                    # if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                    #     tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                    #     error_image = error_map(image, gt_image)
                    #     tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                    #     vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                
                # 构造要写入的内容
                log_line = "[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}\n".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test
                )
                with open(log_file, "a") as f:
                    f.write(log_line)

        #         if tb_writer:
        #             tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
        #             tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        #             tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
        #             tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

