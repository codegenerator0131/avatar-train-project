import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
from src.utils.obj import load_obj
from src.utils.geom import GeometryModule

import math

to_tensor = T.ToTensor()
font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)

class UnNormalize(T.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input range.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)  # Adding a small value to avoid dividing by zero
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())  # Apply denormalization


# Define the denormalization transformation
unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def to_device(batch, device):
    """
    batch: dictionary containing values of various types
    device: target torch device (e.g., "cuda" or "cpu")
    returns: new dictionary with torch.Tensors moved to device
    """
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, dict):
            # 재귀적으로 내부 딕셔너리도 처리
            moved_batch[key] = to_device(value, device)
        elif isinstance(value, list):
            # 리스트 내부 요소가 tensor면 옮기고 아니면 그대로
            moved_batch[key] = [
                v.to(device) if isinstance(v, torch.Tensor) else v for v in value
            ]
        else:
            moved_batch[key] = value  # numpy, int, str 등은 그대로
    return moved_batch

def squeeze_batch_tensors(batch_dict):
    """
    batch 딕셔너리 안의 텐서 값들 중에서
    제일 앞 차원이 1인 텐서에 대해 squeeze(0)를 적용.
    """
    squeezed = {}
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[0] == 1:
            squeezed[k] = v.squeeze(0)
        else:
            squeezed[k] = v
    return squeezed



def imgcat(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).contiguous().squeeze()
        x = x.detach().cpu().numpy()

    print(f'[imgcat] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    import matplotlib.pyplot as plt
    plt.imshow(x)
    plt.savefig('/home/uwang/project/ELITE_WIP/imgcat.png')
    plt.close()




def imgsave(
    input_data,
    save_path,
    make_grid=True,
    text=None,
    text_color="white",
    text_position=(5, 5),
    font_size=20,
    value_range='auto'  # 'auto', '0_1', '0_255', '-1_1'
):
    """
    이미지 텐서 또는 numpy array를 저장하는 함수

    Args:
        input_data (torch.Tensor or np.ndarray): 이미지 데이터 (C,H,W), (B,C,H,W), (N,B,C,H,W)
        save_path (str): 저장 경로
        make_grid (bool): make_grid 적용 여부
        text (str, optional): 이미지 위에 렌더링할 텍스트
        text_color (str or tuple): 텍스트 색상
        text_position (tuple): 텍스트 위치 (x, y)
        font_size (int): 폰트 크기
        value_range (str): 값 범위 조정 방식 ('auto', '0_1', '0_255', '-1_1')
    """

    def to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError("input_data must be a torch.Tensor or np.ndarray")
        return x.float()

    def normalize(t):
        if value_range == '0_1':
            return t.clamp(0, 1)
        elif value_range == '0_255':
            return (t / 255.0).clamp(0, 1)
        elif value_range == '-1_1':
            return ((t + 1) / 2).clamp(0, 1)
        else:  # auto
            if t.max() <= 1.0 and t.min() >= 0.0:
                return t
            elif t.max() <= 255.0:
                return (t / 255.0).clamp(0, 1)
            elif t.min() >= -1.0 and t.max() <= 1.0:
                return ((t + 1) / 2).clamp(0, 1)
            else:
                raise ValueError("Unknown tensor value range. Please specify `value_range` explicitly.")

    # 입력 변환
    tensor = to_tensor(input_data)

    # 차원 정리
    if tensor.dim() == 5:  # (N, B, C, H, W)
        tensor = tensor.view(-1, *tensor.shape[-3:])
    elif tensor.dim() == 3:  # (C, H, W)
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 4:
        raise ValueError("Unsupported input shape. Must be (C,H,W), (B,C,H,W), or (N,B,C,H,W)")

    # 정규화
    tensor = normalize(tensor)

    # make_grid 처리
    if make_grid:
        img = vutils.make_grid(tensor, padding=0)
    else:
        if tensor.shape[0] == 1:
            img = tensor[0]
        else:
            raise ValueError("Multiple images but make_grid=False. Please set make_grid=True.")

    # PIL 이미지로 변환
    pil_img = to_pil_image(img)

    # 텍스트 렌더링
    if text:
        draw = ImageDraw.Draw(pil_img)
        try:
            # font = ImageFont.truetype("arial.ttf", font_size)
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        draw.text(text_position, text, fill=text_color, font=font)

    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 이미지 저장
    pil_img.save(save_path)
    # print(f"Saved image to {save_path}")



def plot_mesh_and_camera(v: torch.Tensor, cam_pos: torch.Tensor, cam_rot: torch.Tensor,
                         elev=30, azim=45, ax=None, cam_scale=0.1, title="Mesh and Camera View Direction"):
    """
    v: [1, N, 3] - SMPL-X mesh vertices
    cam_pos: [1, 3] - camera position
    cam_rot: [1, 3, 3] - camera rotation matrix
    cam_scale: float - length of axis arrows
    """
    v_np = v[0].cpu().numpy()
    cam_pos_np = cam_pos[0].cpu().numpy()
    cam_rot_np = cam_rot[0].cpu().numpy()

    # 카메라 좌표축 (X, Y, Z) - world space 기준 방향
    cam_x = cam_rot_np[:, 0]  # right
    cam_y = cam_rot_np[:, 1]  # up
    cam_z = cam_rot_np[:, 2]  # forward (주의: Z-forward인지 -Z-forward인지 확인 필요)

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # 메시 표시
    ax.scatter(v_np[:, 0], v_np[:, 1], v_np[:, 2], s=0.1, c='blue', alpha=0.5, label='Vertices')

    # 카메라 위치
    ax.scatter(*cam_pos_np, c='red', s=30, label='Camera Position')

    # 카메라 좌표축 시각화
    ax.quiver(*cam_pos_np, *cam_x, length=cam_scale, color='r', label='Camera X (right)')
    ax.quiver(*cam_pos_np, *cam_y, length=cam_scale, color='g', label='Camera Y (up)')
    ax.quiver(*cam_pos_np, *cam_z, length=cam_scale, color='b', label='Camera Z (forward)')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect

    plt.tight_layout()
    plt.show()



def vis_gstex(x, res=512):
    # x: [P, 3]
    P, D = x.shape
    imgcat(x.view(res, res, D).permute(2, 0, 1).contiguous(), renormalize=True)


def psnr(img1, img2):
    # mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


cm = matplotlib.colormaps['Spectral']


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    if len(depth.shape) >= 3:
        depth = depth.squeeze()

    if not isinstance(depth, np.ndarray):
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # Change nan to 0

    if minmax is None:
        mi = np.min(x[x > 0])  # Get minimum positive depth
        ma = np.max(x)  # Get maximum depth
    else:
        mi, ma = minmax

    # Increase the maximum value for the background
    ma_plus = ma + (ma - mi) * 0.1  # Make background 10% more distant than the max depth

    # Assign more distant depth value to background
    x[x == 0] = ma_plus

    # Normalize including the new background value
    x = (x - mi) / (ma_plus - mi + 1e-8)

    x_colored = cm(x, bytes=False)[..., :3]
    x_colored = (255 * x_colored).astype(np.uint8)
    # x = (255 * x).astype(np.uint8)
    # x_colormap = cv2.applyColorMap(x, cmap)

    x_image = Image.fromarray(x_colored)
    x_tensor = T.ToTensor()(x_image)  # Convert to tensor (3, H, W)

    return x_tensor


def visualize_depth_wo_bg(depth, minmax=None, cmap='Spectral'):
    """
    Visualizes the depth map excluding the background (depth of 0), using a specified colormap,
    and makes the background white.
    """
    if len(depth.shape) >= 3:
        depth = depth.squeeze()

    if not isinstance(depth, np.ndarray):
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # Change nan to 0

    # Apply mask to exclude background
    mask = x > 0

    if minmax is None:
        mi = np.min(x[mask])  # Get minimum positive depth
        ma = np.max(x[mask])  # Get maximum depth
    else:
        mi, ma = minmax

    # Initialize an output array with ones (background as white) and multiply by 255 to make it white
    x_colored = np.ones((*x.shape, 3), dtype=np.uint8) * 255

    if np.any(mask):
        # Normalize only non-background pixels
        x_norm = np.zeros_like(x, dtype=float)
        x_norm[mask] = (x[mask] - mi) / (ma - mi + 1e-8)

        # Apply colormap to non-background pixels
        cm = plt.get_cmap(cmap)
        x_colored[mask] = (cm(x_norm[mask])[:, :3] * 255).astype(np.uint8)

    x_image = Image.fromarray(x_colored)
    x_tensor = T.ToTensor()(x_image)  # Convert to tensor (3, H, W)

    return x_tensor


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def compute_face_orientation(verts, faces, return_scale=True):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    # FIXME - this code from GaussianAvatars, but it is wrong.
    # a0 = safe_normalize(v1 - v0)
    # a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    # a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  # will have artifacts without negation
    #
    # orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    # TODO: YW: Modified face orientation code
    # Compute the edge vectors
    a0 = v1 - v0  # [B, F, 3]
    a2 = v2 - v0  # [B, F, 3]

    # Compute the normal vector of each face
    a1 = torch.cross(a0, a2, dim=-1)  # [B, F, 3]
    a1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-20)  # Normalize the normals

    # Create rotation matrices from edge1, edge2, and normal
    # Ensure edge1 and edge2 are orthogonal to normal
    a0 = torch.nn.functional.normalize(a0, dim=-1, eps=1e-20)
    a2 = torch.cross(a1, a0, dim=-1)  # Recompute edge2 to ensure orthogonality
    a2 = torch.nn.functional.normalize(a2, dim=-1, eps=1e-20)

    # Construct the rotation matrix for each face
    orientation = torch.stack([a0, a2, a1], dim=-1)  # [B, F, 3, 3]

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    else:
        scale = None

    return orientation, scale


def compute_unit_normals(verts, faces):
    i0, i1, i2 = faces[..., 0], faces[..., 1], faces[..., 2]

    v0, v1, v2 = verts[..., i0, :], verts[..., i1, :], verts[..., i2, :]

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    unit_normals = safe_normalize(normals)

    return unit_normals


def visualize_normals(points, normals):
    # from mpl_toolkits.mplot3d import Axes3D

    # Example data
    # Points of the mesh (for visualization purpose, replace this with your mesh points)
    # Shape [B, P, 3] - here using a simple example with one batch and 5 points
    # points = np.array([[[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]])
    # # Corresponding unit normal vectors at each point
    # normals = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]]])

    # Select the batch you want to visualize, for example, the first batch
    # Ensure tensors are on CPU and convert them to NumPy arrays
    points_np = points.detach().cpu().numpy()
    normals_np = normals.detach().cpu().numpy()

    # Select the batch you want to visualize, for example, the first batch
    batch_index = 0
    mesh_points = points_np[batch_index]
    mesh_normals = normals_np[batch_index]

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points of the mesh
    ax.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], c=[1.0, 0.0, 0.0], s=0.01)

    # Plot the normal vectors
    for i in range(len(mesh_points)):
        if i < 3000:
            # ax.scatter(mesh_points[i, 0], mesh_points[i, 1], mesh_points[i, 2], s=0.5)
            ax.quiver(mesh_points[i, 0], mesh_points[i, 1], mesh_points[i, 2],
                      mesh_normals[i, 0], mesh_normals[i, 1], mesh_normals[i, 2], length=0.05, normalize=True)

    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-0.25, 0.25])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(azim=270, elev=90)
    plt.show()



import matplotlib.pyplot as plt

def visualize_mesh_difference(original_mesh, deformed_mesh):
    # 변형 벡터 계산
    deformation_vectors = deformed_mesh - original_mesh

    # 메쉬의 vertex 좌표
    X, Y, Z = original_mesh[:, 0], original_mesh[:, 1], original_mesh[:, 2]
    U, V, W = deformation_vectors[:, 0], deformation_vectors[:, 1], deformation_vectors[:, 2]

    # 3D 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 원래 메쉬와 변형된 메쉬의 vertex를 그립니다.
    ax.scatter(original_mesh[:, 0], original_mesh[:, 1], original_mesh[:, 2], color='b', label='Original Mesh', s=0.25)
    # ax.scatter(deformed_mesh[:, 0], deformed_mesh[:, 1], deformed_mesh[:, 2], color='r', label='Deformed Mesh', s=0.25)

    # 변형 벡터를 그립니다.
    # ax.quiver(X, Y, Z, U, V, W, color='g', length=0.1, normalize=True)
    # ax.quiver(X, Y, Z, U, V, W, color='g', length=0.1, normalize=True, linewidths=0.25, arrow_length_ratio=0.1)
    # 변형 벡터를 그립니다.
    for i in range(len(original_mesh)):
        ax.plot([original_mesh[i, 0], deformed_mesh[i, 0]],
                [original_mesh[i, 1], deformed_mesh[i, 1]],
                [original_mesh[i, 2], deformed_mesh[i, 2]],
                color='g', linewidth=0.25)

    ax.view_init(elev=90, azim=270)  # 예시로 설정한 각도입니다.

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def draw_point(points_, color='blue', elev=90, axim=270):
    if isinstance(points_, torch.Tensor):
        points = points_.detach().cpu().numpy()
    else:
        points = points_

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points with size parameter
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=0.05)  # s is the size of points

    ax.view_init(elev=elev, azim=axim)  # 예시로 설정한 각도입니다.
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-0.15, 1.85])
    ax.set_zlim([-1.0, 1.0])

    plt.show()

def plot_flame(points_, color='blue', elev=90, axim=270):
    if isinstance(points_, torch.Tensor):
        points = points_.detach().cpu().numpy()
    else:
        points = points_

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points with size parameter
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=0.05)  # s is the size of points

    ax.view_init(elev=elev, azim=axim)  # 예시로 설정한 각도입니다.
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.2])

    plt.show()

# # Retrieved from GALA
# op_3d_mapping = np.array([
#     24, 12, 17, 19, 21, 16,
#     18, 20, 2, 5, 8, 1,
#     4, 7, 25, 26, 27, 28], dtype=np.int32)

# # smpl2op_npy = np.load('/home/uwang/project/gen_3dgs/data/smpl2openpose.npy')
# op_order = [1, 0, 8, 9, 10, 2, 3, 4, 11, 12, 13, 5, 6, 7, 16, 14, 17, 15]

# import smplx
# def get_canon_body(bm_path, device, return_joints=False):
#     with torch.no_grad():
#         bm = smplx.SMPL(model_path=bm_path, batch_size=1).to(device)
#         canon_pose = torch.zeros([23, 3], dtype=torch.float32, device=device)  # 21
#         # change leg angle a bit
#         canon_pose[0, 2] = 15 * np.pi / 180.
#         canon_pose[1, 2] = -15 * np.pi / 180.
#         canon_pose = canon_pose.view(1, -1)

#         canon_body = bm.forward(body_pose=canon_pose)

#         canon_vert = canon_body.vertices
#         # canon_joint = canon_body.joints[0][op_3d_mapping][None, ...]  # [1, 19, 3]
#         vy_max = torch.max(canon_vert[..., 1])
#         vy_min = torch.min(canon_vert[..., 1])
#         human_height = 1.80
#         smpl_verts = canon_vert / (vy_max - vy_min) * human_height
#         # smpl_joints = canon_joint / (vy_max - vy_min) * human_height
#         min_y = torch.min(smpl_verts[..., 1]).clone()
#         smpl_verts[..., 1] -= min_y
#         if return_joints:
#             smpl2op_npy = np.load('/home/uwang/project/gen_3dgs/data/smpl2openpose.npy')
#             smpl_joints = torch.einsum('...jc,jJ->...Jc', smpl_verts, torch.from_numpy(smpl2op_npy).to(device))[:, op_order]
#         else:
#             smpl_joints = None
#         # smpl_joints[..., 1] -= min_y
#     return smpl_verts, smpl_joints

# def get_canon_body_smplx(bm_path, device, return_joints=False):
#     with torch.no_grad():
#         bm = smplx.SMPLX(model_path=bm_path, batch_size=1, num_pca_comps=12, use_pca=False, flat_hand_mean=False).to(device)
#         canon_pose = torch.zeros([21, 3], dtype=torch.float32, device=device)  # 21
#         # change leg angle a bit
#         canon_pose[0, 2] = 15 * np.pi / 180.
#         canon_pose[1, 2] = -15 * np.pi / 180.
#         canon_pose = canon_pose.view(1, -1)

#         canon_body = bm.forward(body_pose=canon_pose)

#         smpl_verts = canon_body.vertices
#         # canon_joint = canon_body.joints[0][op_3d_mapping][None, ...]  # [1, 19, 3]
#         # vy_max = torch.max(canon_vert[..., 1])
#         # vy_min = torch.min(canon_vert[..., 1])
#         # human_height = 1.80
#         # smpl_verts = canon_vert / (vy_max - vy_min) * human_height
#         # # smpl_joints = canon_joint / (vy_max - vy_min) * human_height
#         # min_y = torch.min(smpl_verts[..., 1]).clone()
#         # smpl_verts[..., 1] -= min_y
#         # if return_joints:
#         #     smpl2op_npy = np.load('/home/uwang/project/gen_3dgs/data/smpl2openpose.npy')
#         #     smpl_joints = torch.einsum('...jc,jJ->...Jc', smpl_verts, torch.from_numpy(smpl2op_npy).to(device))[:, op_order]
#         # else:
#         smpl_joints = None
#         # smpl_joints[..., 1] -= min_y
#     return smpl_verts, smpl_joints


def process_batch(batch, bm, smpl_faces, device):
    with torch.no_grad():
        if len(batch['smpl_body'].shape) == 2:
            smpl_out = bm.forward(
                global_orient=batch['smpl_orient'].to(device),
                body_pose=batch['smpl_body'].to(device),
                # betas=batch['smpl_betas'].to(device),
                betas=torch.zeros_like(batch['smpl_betas'], device=device),
            )
        else:
            smpl_out = bm.forward(
                global_orient=batch['smpl_orient'].squeeze().to(device),
                body_pose=batch['smpl_body'].squeeze().to(device),
                betas=torch.zeros_like(batch['smpl_betas'].squeeze(), device=device),
                # betas=batch['smpl_betas'].squeeze().to(device),
            )
        smpl_verts = smpl_out.vertices

        # compute scale param for alignment
        scan_height = (torch.max(batch['smpl_v'][:, :, 1], -1, keepdim=True)[0] -
                       torch.min(batch['smpl_v'][:, :, 1], -1, keepdim=True)[0]).to(torch.float32)
        smpl_height = (torch.max(smpl_verts[:, :, 1], -1, keepdim=True)[0] -
                       torch.min(smpl_verts[:, :, 1], -1, keepdim=True)[0]).to(torch.float32)
        scale = scan_height.to(device) / smpl_height
        if len(scale.shape) == 2:
            smpl_verts = torch.einsum('b,bmn->bmn', scale.squeeze(1), smpl_verts) + batch['smpl_trans'][:, None, :].to(device)
        else:
            smpl_verts = torch.einsum('b,bmn->bmn', scale.squeeze(), smpl_verts) + batch['smpl_trans'][:, None, :].to(device)

        norm_factor = batch['norm_factor'].to(device)
        # height normalization
        vy_max = norm_factor[:, 0, 0]  # (8,)
        vy_min = norm_factor[:, 0, 1]  # (8,)
        human_height = 1.80
        smpl_verts = smpl_verts / (vy_max - vy_min)[..., None, None] * human_height
        smpld_verts = batch['smpl_v'].to(device) / (vy_max - vy_min)[..., None, None] * human_height

        grounding_y = norm_factor[:, 0, 2][..., None]
        smpl_verts[..., 1] -= grounding_y  # "ground" each mesh's feet onto xz plane
        smpld_verts[..., 1] -= grounding_y
        # triangles = smpl_verts[:, smpl_faces]  # coordinates of mesh triangles  [Bs, 20908, 3, 3]
        # gs_xyz = triangles.mean(dim=-2).squeeze(0)  # center coordinate of mesh triangles  [Bs, 20908, 3]
        # scan_xyz = scan_verts[:, smpl_faces].mean(dim=-2).squeeze(0)  # center coordinate of mesh triangles  [Bs, 20908, 3]
        # face_orien_mat, face_scale = compute_face_orientation(smpl_verts, smpl_faces, return_scale=True)  # face normal direction vector [Bs, 20908, 3, 3]
        # face_orien_mat_canon, face_scale_canon = compute_face_orientation(canon_vert, smpl_faces, return_scale=True)
        # face_scale /= face_scale_canon
        # face_orien_quat = rotmat_to_unitquat(face_orien_mat)  # Quaternion, xyzw representation : default in Roma library

    # return gs_xyz, scan_xyz, face_orien_quat, face_orien_mat, face_scale
    # return gs_xyz, smpl_verts
    return smpl_verts, smpld_verts


def get_flame_expr(batch, fm):
    with torch.no_grad():
        # for driving signal
        verts_expr, _ = fm.forward(
            shape=torch.zeros_like(batch['flame_shape']),
            expr=batch['flame_expr'],
            rotation=torch.zeros_like(batch['flame_rot']),
            neck=torch.zeros_like(batch['flame_neck']),
            jaw=batch['flame_jaw'],
            eyes=batch['flame_eyes'],
            translation=torch.zeros_like(batch['flame_trans']),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
        # for canon visualization
        _, verts_neut = fm.forward(
            shape=torch.zeros_like(batch['flame_shape']),
            expr=torch.zeros_like(batch['flame_expr']),
            rotation=torch.zeros_like(batch['flame_rot']),
            neck=torch.zeros_like(batch['flame_neck']),
            jaw=torch.zeros_like(batch['flame_jaw']),
            eyes=torch.zeros_like(batch['flame_eyes']),
            translation=torch.zeros_like(batch['flame_trans']),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
    return verts_expr, verts_neut

def process_flame(batch, fm, use_shape_betas=False):
    if use_shape_betas:
        betas = batch['flame_shape']
    else:
        betas = torch.zeros_like(batch['flame_shape'])
    with torch.no_grad():
        # for driving signal
        verts_posed, _ = fm.forward(
            shape=betas,
            expr=batch['flame_expr'],
            rotation=batch['flame_rot'],
            neck=batch['flame_neck'],
            jaw=batch['flame_jaw'],
            eyes=batch['flame_eyes'],
            translation=batch['flame_trans'],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
        # for canon visualization
        _, verts_canon = fm.forward(
            shape=betas,
            expr=torch.zeros_like(batch['flame_expr']),
            rotation=torch.zeros_like(batch['flame_rot']),
            neck=torch.zeros_like(batch['flame_neck']),
            jaw=torch.zeros_like(batch['flame_jaw']),
            eyes=torch.zeros_like(batch['flame_eyes']),
            translation=torch.zeros_like(batch['flame_trans']),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
        if 'flame_stoffset' in batch.keys():
            # for mesh geometry encoding
            _, enc_verts_cano = fm.forward(
                shape=batch['flame_shape'],
                expr=torch.zeros_like(batch['flame_expr']),
                rotation=torch.zeros_like(batch['flame_rot']),
                neck=torch.zeros_like(batch['flame_neck']),
                jaw=torch.zeros_like(batch['flame_jaw']),
                eyes=torch.zeros_like(batch['flame_eyes']),
                translation=torch.zeros_like(batch['flame_trans']),
                zero_centered_at_root_node=True,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=batch['flame_stoffset'],
                dynamic_offset=None,
            )
        else:
            enc_verts_cano = None
    return verts_posed, verts_canon, enc_verts_cano

def process_enc_flame(batch, fm):
    with torch.no_grad():
        # for mesh geometry encoding
        _, enc_verts_cano = fm.forward(
            shape=batch['enc_flame_shape'],
            expr=torch.zeros_like(batch['flame_expr']),
            rotation=torch.zeros_like(batch['flame_rot']),
            neck=torch.zeros_like(batch['flame_neck']),
            jaw=torch.zeros_like(batch['flame_jaw']),
            eyes=torch.zeros_like(batch['flame_eyes']),
            translation=torch.zeros_like(batch['flame_trans']),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=batch['enc_flame_stoffset'],
            dynamic_offset=None,
        )
    return enc_verts_cano

def process_canon_mean_flame(batch, fm):
    with torch.no_grad():
        # mean shape, mean pose/expr mesh
        _, verts_cano_mean = fm.forward(
            shape=torch.zeros_like(batch['flame_shape']),
            expr=torch.zeros_like(batch['flame_expr']),
            rotation=torch.zeros_like(batch['flame_rot']),
            neck=torch.zeros_like(batch['flame_neck']),
            jaw=torch.zeros_like(batch['flame_jaw']),
            eyes=torch.zeros_like(batch['flame_eyes']),
            translation=torch.zeros_like(batch['flame_trans']),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=torch.zeros_like(batch['flame_stoffset']),
            dynamic_offset=None,
        )
    return verts_cano_mean


def process_flame_difix_input_img(batch, fm):
    betas = torch.zeros_like(batch['input_flame_shape'])
    with torch.no_grad():
        # for driving signal
        verts_posed, _ = fm.forward(
            shape=betas,
            expr=batch['input_flame_expr'],
            rotation=batch['input_flame_rot'],
            neck=batch['input_flame_neck'],
            jaw=batch['input_flame_jaw'],
            eyes=batch['input_flame_eyes'],
            translation=batch['input_flame_trans'],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
        # for canon visualization
        _, verts_canon = fm.forward(
            shape=betas,
            expr=torch.zeros_like(batch['input_flame_expr']),
            rotation=torch.zeros_like(batch['input_flame_rot']),
            neck=torch.zeros_like(batch['input_flame_neck']),
            jaw=torch.zeros_like(batch['input_flame_jaw']),
            eyes=torch.zeros_like(batch['input_flame_eyes']),
            translation=torch.zeros_like(batch['input_flame_trans']),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
    return verts_posed, verts_canon, None


def process_flame_seq(batch, fm, device):
    all_verts_posed = []
    with torch.no_grad():
        N = batch['flame_trans'].shape[0]
        T = batch['flame_trans'].shape[1]
        # batch['flame_trans'].shape == [N, T, 3]
        for t in range(T):
            # for driving signal
            verts_posed, _ = fm.forward(
                shape=torch.zeros_like(batch['flame_shape']),
                expr=batch['flame_expr'][:,t],
                rotation=batch['flame_rot'][:, t],
                neck=batch['flame_neck'][:, t],
                jaw=batch['flame_jaw'][:, t],
                eyes=batch['flame_eyes'][:, t],
                translation=batch['flame_trans'][:, t],
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=None,
                dynamic_offset=None,
            )
            all_verts_posed.append(verts_posed)

        # stack in time axis  [B, T, 5143, 3]
        all_verts_posed = torch.stack(all_verts_posed, dim=1)
        # for mesh geometry encoding
        _, enc_verts_cano = fm.forward(
            shape=batch['flame_shape'],
            expr=torch.zeros_like(batch['flame_expr'][:, 0]),
            rotation=torch.zeros_like(batch['flame_rot'][:, 0]),
            neck=torch.zeros_like(batch['flame_neck'][:, 0]),
            jaw=torch.zeros_like(batch['flame_jaw'][:, 0]),
            eyes=torch.zeros_like(batch['flame_eyes'][:, 0]),
            translation=torch.zeros_like(batch['flame_trans'][:, 0]),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=batch['flame_stoffset'],
            dynamic_offset=None,
        )
        # for canon visualization
        _, verts_canon = fm.forward(
            shape=torch.zeros_like(batch['flame_shape']),
            expr=torch.zeros_like(batch['flame_expr'][:, 0]),
            rotation=torch.zeros_like(batch['flame_rot'][:, 0]),
            neck=torch.zeros_like(batch['flame_neck'][:, 0]),
            jaw=torch.zeros_like(batch['flame_jaw'][:, 0]),
            eyes=torch.zeros_like(batch['flame_eyes'][:, 0]),
            translation=torch.zeros_like(batch['flame_trans'][:, 0]),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
    return all_verts_posed, verts_canon, enc_verts_cano

def process_flame_vae(batch, fm, device):
    with torch.no_grad():
        # for driving signal
        verts_posed, _ = fm.forward(
            shape=torch.zeros_like(batch['flame_shape']),
            expr=batch['flame_expr'],
            rotation=batch['flame_rot'],
            neck=batch['flame_neck'],
            jaw=batch['flame_jaw'],
            eyes=batch['flame_eyes'],
            translation=batch['flame_trans'],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
        # for mesh geometry encoding
        _, enc_verts_cano = fm.forward(
            shape=batch['flame_shape'],
            expr=torch.zeros_like(batch['flame_expr']),
            rotation=torch.zeros_like(batch['flame_rot']),
            neck=torch.zeros_like(batch['flame_neck']),
            jaw=torch.zeros_like(batch['flame_jaw']),
            eyes=torch.zeros_like(batch['flame_eyes']),
            translation=torch.zeros_like(batch['flame_trans']),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=batch['flame_stoffset'],
            dynamic_offset=None,
        )
        # for canon visualization
        _, verts_canon = fm.forward(
            shape=torch.zeros_like(batch['flame_shape']),
            expr=torch.zeros_like(batch['flame_expr']),
            rotation=torch.zeros_like(batch['flame_rot']),
            neck=torch.zeros_like(batch['flame_neck']),
            jaw=torch.zeros_like(batch['flame_jaw']),
            eyes=torch.zeros_like(batch['flame_eyes']),
            translation=torch.zeros_like(batch['flame_trans']),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
    return verts_posed, verts_canon, enc_verts_cano



def get_canon_flame(fm, device):
    with torch.no_grad():
        verts, verts_cano, lmks = fm.forward(
            shape=torch.zeros((1, 300), device=device),
            expr=torch.zeros((1, 100), device=device),
            rotation=torch.zeros((1, 3), device=device),
            neck=torch.zeros((1, 3), device=device),
            jaw=torch.zeros((1, 3), device=device),
            eyes=torch.zeros((1, 6), device=device),
            translation=torch.zeros((1, 3), device=device),
            zero_centered_at_root_node=False,
            return_landmarks=True,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
    return verts, verts_cano, lmks


def load_geo_fn_flame(flame, cfg, device):
    geo_fn = GeometryModule(
        flame.faces.cpu(),
        flame.verts_uvs.cpu(),
        flame.textures_idx.cpu(),
        None,
        v=flame.v_template.cpu(),
        uv_size=cfg['uv_size'],
        # flip_uv=False,
        flip_uv=True,
        impaint=True,
        # impaint=False,
    ).to(device)
    return geo_fn


def resize_transform(image, target_size=(224, 224), fill_color=1):
    # image: Tensor shape (3, H, W)
    b, c, h, w = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resize_tf = T.Resize((new_h, new_w))

    resized = resize_tf(image)
    pad_top = (target_size[0] - new_h) // 2
    pad_left = (target_size[1] - new_w) // 2
    pad_bottom = target_size[0] - new_h - pad_top
    pad_right = target_size[1] - new_w - pad_left

    pad_tf = T.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=fill_color)
    padded = pad_tf(resized)
    return padded



if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    # 더미 모델 & 옵티마이저
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)  # placeholder

    # 스케줄러 설정
    total_epochs = 10001
    T_up = 50
    start_lr = 1e-6
    init_lr = 0.0002
    final_lr = 1e-5

    scheduler = WarmupCosineDecayLR(
        optimizer,
        total_epochs=total_epochs,
        start_lr=start_lr,
        init_lr=init_lr,
        final_lr=final_lr,
        T_up=T_up
    )

    # 학습률 기록
    lrs = []
    for epoch in range(total_epochs):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # 그래프 그리기
    plt.figure(figsize=(8, 4))
    plt.plot(range(total_epochs), lrs, marker='o')
    plt.title("Warm-up + Cosine Decay LR Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()