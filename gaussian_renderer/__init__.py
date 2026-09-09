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
import math
from typing import Union
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh

from pytorch3d.structures import Meshes

from utils.general_utils import quatProduct_batch

def render_normal(flame_model, viewpoint_cam, gaussians, pipe, background, asset):
        # world -> camera
        R_tensor = torch.from_numpy(viewpoint_cam.R).float().cuda()
        T_tensor = torch.from_numpy(viewpoint_cam.T).float().cuda()
        xyz = torch.matmul(R_tensor, asset['mean_3d'].permute(1,0)).permute(1,0) + T_tensor.view(1,3)

        normal = Meshes(
            verts=xyz[None], 
            faces=torch.LongTensor(flame_model.face_upsampled).cuda()[None]
        ).verts_normals_packed().reshape(flame_model.vertex_num_upsampled, 3)

        normal = torch.stack((normal[:,0], -normal[:,1], -normal[:,2]), 1)  # flip y,z

        # asset_normal = {k: normal if k == 'rgb' else v for k,v in asset.items()}
        asset_normal = {**asset, 'rgb': normal}
        normal = render2(
            viewpoint_cam, gaussians, pipe, background, asset_normal
        )['render']
        normal = (normal + 1) / 2.  # [0,1]
        return normal

# def render2(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor, gaussian_assets = None, scaling_modifier = 1.0, override_color = None):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
    
#     screenspace_points = torch.zeros_like(gaussian_assets['mean_3d'], dtype=gaussian_assets['mean_3d'].dtype, requires_grad=True, device="cuda") + 0
    
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform.cuda(),
#         projmatrix=viewpoint_camera.full_proj_transform.cuda(),
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center.cuda(),
#         prefiltered=False,
#         debug=pipe.debug
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
#     means3D = gaussian_assets['mean_3d'] 
#     means2D = screenspace_points
#     opacity = gaussian_assets['opacity']   # 不透明度 (N, 1)
#     rgb = gaussian_assets['rgb'] 
#     scales = pc.get_scaling_mesh
#     rotations = gaussian_assets['rotation']
       

#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = None,
#         colors_precomp = rgb,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = None)

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : radii > 0,
#             "radii": radii}



def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor, gaussian_assets = None, offset: torch.Tensor = None, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if not pc.stage2_is_on:
        screenspace_points = torch.zeros_like(gaussian_assets['mean_3d'], dtype=gaussian_assets['mean_3d'].dtype, requires_grad=True, device="cuda") + 0
    else:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if not pc.stage2_is_on:
        means3D = gaussian_assets['mean_3d'] 
        means2D = screenspace_points
        opacity = gaussian_assets['opacity']   # 不透明度 (N, 1)
    else:
        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if not pc.stage2_is_on:
            scales = pc.get_scaling_mesh
            rotations = gaussian_assets['rotation']
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if not pc.stage2_is_on:
                shs = pc.get_features_mesh
            else:
                # shs = pc.get_features
                # xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
                # dir_pp = means3D - viewpoint_camera.camera_center.to(means3D.device).repeat(means3D.shape[0], 1)

                # dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # shs = pc.mlp_head(torch.cat([pc.recolor(xyz), pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
                # shs = shs.float()
                with torch.no_grad():
                    means3D_cano = pc.get_xyz_cano
                xyz = pc.contract_to_unisphere(means3D_cano.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
                dir_pp = means3D_cano - viewpoint_camera.camera_center.to(means3D_cano.device).repeat(means3D_cano.shape[0], 1)

                dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                shs = pc.mlp_head(torch.cat([pc.recolor(xyz), pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
                shs = shs.float()

    else:
        colors_precomp = override_color


    #========= Offset ==========
    if offset is not None:
        offset = offset.squeeze(0)
        means3D += offset[:, :3]
        scales *= offset[:, 3:6]
        rotations = quatProduct_batch(offset[:, 6:10], rotations)
        shs[:, 0, :3] += offset[:, 10:13]
        opacity = torch.clamp(opacity + offset[:, 13, None], 0, 1)


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

# def render2(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform.cuda(),
#         projmatrix=viewpoint_camera.full_proj_transform.cuda(),
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center.cuda(),
#         prefiltered=False,
#         debug=pipe.debug
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = pc.get_features
#     else:
#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : radii > 0,
#             "radii": radii}
