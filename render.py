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

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer


mesh_renderer = NVDiffRenderer()

import numpy as np
import torch
import trimesh 

from networks.dual_branch import DualBranchUNet
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


def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")


#原始
# def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh):
#     if dataset.select_camera_id != -1:  # 1. 如果指定了 select_camera_id，则修改输出名称
#         name = f"{name}_{dataset.select_camera_id}" # 在名字后追加 camera id，方便区分不同相机渲染结果
#     iter_path = Path(dataset.model_path) / name / f"ours_{iteration}" # 2. 构造输出路径结构
#     render_path = iter_path / "renders" # 渲染结果路径
#     gts_path = iter_path / "gt" # GT（ground truth）路径

# #     render_offset_path = iter_path / "offset_map"
# #     render_offset_maps(
# #     gaussians.dual_branch_net.uv_sample_coords,
# #     offset,
# #     render_offset_path,
# #     uv_size=256,
# # )
#     if render_mesh: # 如果需要渲染mesh结果
#         render_mesh_path = iter_path / "renders_mesh"
#         makedirs(render_mesh_path, exist_ok=True)

#     # render_mesh_path = iter_path / "renders_mesh"
#     # makedirs(render_mesh_path, exist_ok=True)

#     makedirs(render_path, exist_ok=True) # 创建目录（如果不存在则创建）
#     makedirs(gts_path, exist_ok=True)

#     views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8) # 3. 数据加载器（逐帧读取view） # DataLoader用于按顺序遍历所有相机视角
#     max_threads = multiprocessing.cpu_count() # 获取CPU核心数（用于控制线程并行）
#     print('Max threads: ', max_threads)
#     worker_args = [] # 用于暂存线程任务
#     for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")): # 4. 主渲染循环（逐view渲染）
#         gaussians.select_mesh_by_timestep_render(view.timestep)
#         displacement_map = gaussians.get_vertex_displace_map()
#         offset = gaussians.dual_branch_net(gaussians.flame_param, view.timestep, displacement_map)
#         render_pkg = render(view, gaussians, pipeline, background, offset=offset)
#         # render_pkg = render(view, gaussians, pipeline, background)
#         rendering = render_pkg["render"]
#         gt = view.original_image[0:3, :, :]
#         if render_mesh: # 4.4 如果需要渲染mesh（额外可视化）
#             # out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
#             # rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
#             # rgb_mesh = rgba_mesh[:3, :, :]
#             # alpha_mesh = rgba_mesh[3:, :, :]
#             # mesh_opacity = 0.5
#             # rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
            
#             out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
#             rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
#             rgb_mesh = rgba_mesh[:3, :, :]
#             alpha_mesh = rgba_mesh[3:, :, :]
#             background = torch.ones_like(rgb_mesh)  # 白底
#             rendering_mesh = rgb_mesh * alpha_mesh + background * (1 - alpha_mesh)
        
#         # save_mesh(gaussians.verts.squeeze(0), gaussians.faces, render_mesh_path / f'{idx:05d}.obj')
#         path2data = {} # 5. 构造写文件任务
#         path2data[Path(render_path) / f'{idx:05d}.png'] = rendering  # 保存render结果
#         path2data[Path(gts_path) / f'{idx:05d}.png'] = gt  # 保存GT
#         if render_mesh: # 保存mesh渲染（如果有）
#             path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
#         worker_args.append([path2data])  # 将当前帧写文件任务加入队列

#         if len(worker_args) == max_threads or idx == len(views_loader)-1:  # 6. 多线程批量写文件
#             with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
#                 futures = [executor.submit(write_data, *args) for args in worker_args]
#                 concurrent.futures.wait(futures)
#             worker_args = []
    
#     try: # 7. 用ffmpeg生成视频
#         os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders.mp4")
#         os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4")
#         if render_mesh:
#             os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
#     except Exception as e:
#         print(e)

import time
#测试render fps
def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh):

    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"

    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"


    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"
        makedirs(render_mesh_path, exist_ok=True)


    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)


    views_loader = DataLoader(
        views,
        batch_size=None,
        shuffle=False,
        num_workers=8
    )


    max_threads = multiprocessing.cpu_count()

    print('Max threads: ', max_threads)

    worker_args = []


    # =====================================================
    # Accurate pure rendering FPS measurement
    # =====================================================

    render_time_total = 0.0
    render_frame_num = 0



    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):


        gaussians.select_mesh_by_timestep_render(view.timestep)

        displacement_map = gaussians.get_vertex_displace_map()

        offset = gaussians.dual_branch_net(
            gaussians.flame_param,
            view.timestep,
            displacement_map
        )


        # =====================================================
        # Only measure Gaussian rendering time
        # =====================================================

        if torch.cuda.is_available():

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()

            starter.record()


            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                offset=offset
            )


            ender.record()

            torch.cuda.synchronize()


            elapsed_time = starter.elapsed_time(ender)  # ms

            render_time_total += elapsed_time

            render_frame_num += 1


        else:

            start_time = time.time()

            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                offset=offset
            )

            end_time = time.time()

            render_time_total += (
                (end_time - start_time) * 1000
            )

            render_frame_num += 1



        # =====================================================
        # Original rendering pipeline
        # =====================================================


        rendering = render_pkg["render"]

        gt = view.original_image[0:3, :, :]


        if render_mesh:

            out_dict = mesh_renderer.render_from_camera(
                gaussians.verts,
                gaussians.faces,
                view
            )

            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)

            rgb_mesh = rgba_mesh[:3, :, :]

            alpha_mesh = rgba_mesh[3:, :, :]

            background = torch.ones_like(rgb_mesh)

            rendering_mesh = (
                rgb_mesh * alpha_mesh
                +
                background * (1 - alpha_mesh)
            )


        path2data = {}


        path2data[
            Path(render_path) / f'{idx:05d}.png'
        ] = rendering


        path2data[
            Path(gts_path) / f'{idx:05d}.png'
        ] = gt



        if render_mesh:

            path2data[
                Path(render_mesh_path) / f'{idx:05d}.png'
            ] = rendering_mesh



        worker_args.append([path2data])



        if len(worker_args) == max_threads or idx == len(views_loader)-1:

            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:

                futures = [
                    executor.submit(write_data, *args)
                    for args in worker_args
                ]

                concurrent.futures.wait(futures)


            worker_args = []



    # =====================================================
    # Print pure rendering FPS
    # =====================================================

    if render_frame_num > 0:

        avg_render_time = (
            render_time_total / render_frame_num
        )

        fps = 1000.0 / avg_render_time


        print("\n================================")
        print("Pure Gaussian Rendering FPS")
        print("================================")
        print(
            f"Frames measured: {render_frame_num}"
        )
        print(
            f"Average render time: {avg_render_time:.4f} ms/frame"
        )
        print(
            f"FPS: {fps:.2f}"
        )
        print("================================\n")



    try:

        os.system(
            f"ffmpeg -y -framerate 25 "
            f"-f image2 -pattern_type glob "
            f"-i '{render_path}/*.png' "
            f"-pix_fmt yuv420p "
            f"{iter_path}/renders.mp4"
        )


        os.system(
            f"ffmpeg -y -framerate 25 "
            f"-f image2 -pattern_type glob "
            f"-i '{gts_path}/*.png' "
            f"-pix_fmt yuv420p "
            f"{iter_path}/gt.mp4"
        )


        if render_mesh:

            os.system(
                f"ffmpeg -y -framerate 25 "
                f"-f image2 -pattern_type glob "
                f"-i '{render_mesh_path}/*.png' "
                f"-pix_fmt yuv420p "
                f"{iter_path}/renders_mesh.mp4"
            )


    except Exception as e:

        print(e)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool):
    with torch.no_grad(): # 不计算梯度（推理/渲染模式）
        if dataset.bind_to_mesh:
            gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False) # 2. 创建场景对象  # Scene负责加载相机、数据集以及训练/测试划分
        
        gaussians.stage2_is_on = True
        if gaussians.binding is None:
            gaussians.binding = torch.arange(len(gaussians.flame_model.faces)).cuda()
            gaussians.binding_counter = torch.ones(len(gaussians.flame_model.faces), dtype=torch.int32).cuda()
        gaussians.create_from_pcd(gaussians.cameras_extent)
        #加载测试文件
        ckpt = torch.load(os.path.join(args.model_path, f"chkpnt{args.iteration}.pth")) # 3. 加载checkpoint # 从磁盘读取训练保存的模型
        (model_capture, saved_iteration) = ckpt # checkpoint结构拆包
        gaussians_state_dict, _ = model_capture
        gaussians._xyz = torch.nn.Parameter(torch.empty_like(gaussians_state_dict["_xyz"]))
        gaussians._scaling = torch.nn.Parameter(torch.empty_like(gaussians_state_dict["_scaling"]))
        gaussians._rotation = torch.nn.Parameter(torch.empty_like(gaussians_state_dict["_rotation"]))
        gaussians._opacity = torch.nn.Parameter(torch.empty_like(gaussians_state_dict["_opacity"]))  
        gaussians.binding = gaussians_state_dict["binding"].clone()
        gaussians.binding_counter = gaussians_state_dict["binding_counter"].clone()    
        gaussians.xyz_cano = gaussians_state_dict["xyz_cano"].clone()  

        cano_position = gaussians.xyz_cano[:gaussians.flame_model.vertex_num, :].clone().unsqueeze(0)
        position_map = gaussians.get_position_map(cano_position)
        uv_mask = gaussians.get_uv_mask()
        uv_coords = torch.load(os.path.join(args.model_path, f"uv_coords{args.iteration}.pt"))

        gaussians.dual_branch_net = DualBranchUNet(
            device="cuda",
            uv_sample_coords=uv_coords,
            uv_mask=uv_mask,
            reference_image=None,
            position_map=position_map,
        )
        gaussians.dual_branch_net.to("cuda")
        
        gaussians.load_state_dict(gaussians_state_dict) # 将保存的state_dict加载回当前模型
        gaussians.get_cano_face_info(gaussians.xyz_cano[:gaussians.flame_model.vertex_num, :].unsqueeze(0))

      

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0] # 5. 设置背景颜色
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 转换为GPU tensor

        if dataset.target_path != "": # 6. 判断是否使用target_path模式
             name = os.path.basename(os.path.normpath(dataset.target_path))
             # when loading from a target path, test cameras are merged into the train cameras
             render_set(dataset, f'{name}', args.iteration, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh)
        else: # 7. 正常模式：分别渲染 train / val / test
            if not skip_train: # 如果不跳过训练集渲染
                render_set(dataset, "train", args.iteration, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh)
            
            if not skip_val: # 如果不跳过验证集渲染
                render_set(dataset, "val", args.iteration, scene.getValCameras(), gaussians, pipeline, background, render_mesh)

            if not skip_test: # 如果不跳过测试集渲染
                render_set(dataset, "test", args.iteration, scene.getTestCameras(), gaussians, pipeline, background, render_mesh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh)


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


import os
import cv2
import numpy as np
import torch


def build_uv_map(feature, u, v, uv_size=256):
    """
    feature: [N,C]
    u,v: [N]
    return: [C,H,W]
    """

    C = feature.shape[1]
    device = feature.device

    uv_map = torch.zeros(
        C,
        uv_size,
        uv_size,
        device=device
    )

    count = torch.zeros(
        1,
        uv_size,
        uv_size,
        device=device
    )

    # 累加特征
    for c in range(C):
        uv_map[c].index_put_(
            (v, u),
            feature[:, c],
            accumulate=True
        )

    # 累加计数
    count[0].index_put_(
        (v, u),
        torch.ones_like(v, dtype=torch.float32),
        accumulate=True
    )

    uv_map = uv_map / (count + 1e-8)

    return uv_map


def save_uv_map(uv_map, save_path):
    """
    uv_map: [C,H,W]
    """

    uv_map = uv_map.detach().cpu()

    if uv_map.shape[0] == 1:

        img = uv_map[0].numpy()

        img = (
            img - img.min()
        ) / (
            img.max() - img.min() + 1e-8
        )

        img = (img * 255).astype(np.uint8)

    else:

        img = uv_map[:3].permute(1, 2, 0).numpy()

        img = (
            img - img.min()
        ) / (
            img.max() - img.min() + 1e-8
        )

        img = (img * 255).astype(np.uint8)

        # RGB -> BGR
        img = img[..., ::-1]

    cv2.imwrite(save_path, img)


def render_offset_maps(
    uv_sample_coords,
    offset,
    render_offset_path,
    uv_size=256,
):

    os.makedirs(render_offset_path, exist_ok=True)

    uv = uv_sample_coords.squeeze(0)   # [N,2]
    offset = offset.squeeze(0)         # [N,14]

    # --------------------------------------------------
    # UV坐标 -> 图像像素坐标
    # 修正上下翻转
    # --------------------------------------------------

    u = (uv[:, 0] * (uv_size - 1)).long()

    v = (
        (1.0 - uv[:, 1]) * (uv_size - 1)
    ).long()

    u = torch.clamp(u, 0, uv_size - 1)
    v = torch.clamp(v, 0, uv_size - 1)

    # --------------------------------------------------
    # xyz
    # --------------------------------------------------
    xyz_map = build_uv_map(
        offset[:, 0:3],
        u,
        v,
        uv_size,
    )
    # --------------------------------------------------
    # scale
    # --------------------------------------------------
    scale_map = build_uv_map(
        offset[:, 3:6],
        u,
        v,
        uv_size,
    )
    # --------------------------------------------------
    # quaternion
    # --------------------------------------------------
    rot_map = build_uv_map(
        offset[:, 6:10],
        u,
        v,
        uv_size,
    )
    # --------------------------------------------------
    # rgb/sh
    # --------------------------------------------------
    rgb_map = build_uv_map(
        offset[:, 10:13],
        u,
        v,
        uv_size,
    )
    # --------------------------------------------------
    # opacity
    # --------------------------------------------------
    opacity_map = build_uv_map(
        offset[:, 13:14],
        u,
        v,
        uv_size,
    )
    # --------------------------------------------------
    # 保存
    # --------------------------------------------------
    save_uv_map(
        xyz_map,
        os.path.join(
            render_offset_path,
            "xyz_offset.png"
        )
    )
    save_uv_map(
        scale_map,
        os.path.join(
            render_offset_path,
            "scale_offset.png"
        )
    )
    # quaternion只显示前三维
    save_uv_map(
        rot_map[:3],
        os.path.join(
            render_offset_path,
            "rot_offset.png"
        )
    )
    save_uv_map(
        rgb_map,
        os.path.join(
            render_offset_path,
            "rgb_offset.png"
        )
    )
    save_uv_map(
        opacity_map,
        os.path.join(
            render_offset_path,
            "opacity_offset.png"
        )
    )
    print(
        f"Offset maps saved to {render_offset_path}"
    )