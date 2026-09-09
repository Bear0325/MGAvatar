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
from copy import deepcopy
import random
import json
from typing import Union, List
import numpy as np
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.flame_gaussian_model import FlameGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.general_utils import PILtoTorch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path

class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        # if isinstance(idx, int):
        #     # ---- from readCamerasFromTransforms() ----
        #     camera = deepcopy(self.cameras[idx])

        #     if camera.image is None:
        #         image = Image.open(camera.image_path)
        #     else:
        #         image = camera.image

        #     im_data = np.array(image.convert("RGBA"))
        #     norm_data = im_data / 255.0
        #     arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + camera.bg * (1 - norm_data[:, :, 3:4])
        #     image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        #     # ---- from loadCam() and Camera.__init__() ----
        #     resized_image_rgb = PILtoTorch(image, (camera.image_width, camera.image_height))

        #     image = resized_image_rgb[:3, ...]

        #     if resized_image_rgb.shape[1] == 4:
        #         gt_alpha_mask = resized_image_rgb[3:4, ...]
        #         image *= gt_alpha_mask
            
        #     camera.original_image = image.clamp(0.0, 1.0)
        #     return camera


        #=================对于没有提前mask掉的数据集========================
        if isinstance(idx, int):
            # ---- from readCamerasFromTransforms() ----
            camera = deepcopy(self.cameras[idx])

            if camera.image is None:
                image = Image.open(camera.image_path)
            else:
                image = camera.image

            # im_data = np.array(image.convert("RGBA"))
            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + camera.bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image_path = Path(camera.image_path)
            mask_path = image_path.parents[1] / "masks" / image_path.name
            mask = Image.open(mask_path)
            image = np.array(image)
            mask  = np.repeat(np.array(mask)[:,:,None], 3, axis=2) / 255.0

            norm_data = image / 255.0
            arr = norm_data * mask + camera.bg * (1-mask)
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            

            # ---- from loadCam() and Camera.__init__() ----
            resized_image_rgb = PILtoTorch(image, (camera.image_width, camera.image_height))

            image = resized_image_rgb[:3, ...]

            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                image *= gt_alpha_mask
            
            camera.original_image = image.clamp(0.0, 1.0)
            return camera
            
        elif isinstance(idx, slice):
            return CameraDataset(self.cameras[idx])
        else:
            raise TypeError("Invalid argument type")

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : Union[GaussianModel, FlameGaussianModel], load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        # 1. 初始化基本属性
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # 2. 如果指定了加载迭代次数，则设置 loaded_iter
        if load_iteration:
            if load_iteration == -1: # 自动查找最大迭代次数
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 3. 加载数据集
        # load dataset
        assert os.path.exists(args.source_path), "Source path does not exist: {}".format(args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):  # 判断数据集类型
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval) # COLMAP 稀疏重建
        elif os.path.exists(os.path.join(args.source_path, "canonical_flame_param.npz")):
            print("Found FLAME parameter, assuming dynamic NeRF data set!") 
            scene_info = sceneLoadTypeCallbacks["DynamicNerf"](args.source_path, args.white_background, args.eval, target_path=args.target_path)  # 动态 NeRF 数据集（FLAME 参数存在）
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)  # Blender 渲染数据集
        else:
            assert False, "Could not recognize scene type!"  # 数据集类型不识别
        
        # 4. 处理相机信息
        # process cameras
        self.train_cameras = {}
        self.val_cameras = {}
        self.test_cameras = {}
        
        if not self.loaded_iter:  
            # if gaussians.binding == None: # 复制 ply 文件（如果高斯点未绑定）
            #     with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #         dest_file.write(src_file.read())
            # 将训练/验证/测试相机保存为 JSON
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.val_cameras:
                camlist.extend(scene_info.val_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        
        # 5. 随机打乱相机顺序
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.val_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.gaussians.cameras_extent = scene_info.nerf_normalization["radius"]  # 6. 相机归一化范围（半径）

        # 7. 创建不同分辨率的训练/验证/测试相机列表
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Validation Cameras")
            self.val_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.val_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        with torch.no_grad():
            self.gaussians.init(scene_info.train_meshes, scene_info.test_meshes, 
                                    scene_info.tgt_train_meshes, scene_info.tgt_test_meshes) 
            
        # 8. 加载网格（如果高斯点绑定了面片）
        # process meshes 
        # if gaussians.binding != None:
        #     self.gaussians.load_meshes(scene_info.train_meshes, scene_info.test_meshes, 
        #                                scene_info.tgt_train_meshes, scene_info.tgt_test_meshes)
        # # 9. 创建或加载高斯点
        # # create gaussians
        # if self.loaded_iter:
        #     self.gaussians.load_ply( # 加载已训练的点云 ply 文件
        #         os.path.join(self.model_path,
        #                     "point_cloud",
        #                     "iteration_" + str(self.loaded_iter),
        #                     "point_cloud.ply"),
        #         has_target=args.target_path != "",
        #     )
        # else:
        #     self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)  # 从点云创建高斯点

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale])
    
    def getValCameras(self, scale=1.0):
        return CameraDataset(self.val_cameras[scale])

    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale])