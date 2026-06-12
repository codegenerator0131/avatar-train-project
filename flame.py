"""
FLAME 2023 forward model.

Loads flame2023.pkl and exposes a differentiable forward pass:
  vertices, landmarks = FLAME(shape, expression, pose, jaw_pose)

Expected pkl keys (FLAME 2023):
  v_template      (5023, 3)   mean neutral mesh
  shapedirs       (5023*3, 300) shape PCA basis  [or (5023, 3, 300)]
  expressiondirs  (5023*3, 100) expression PCA basis
  posedirs        (5023*3, 36)  pose corrective blendshapes (9 joints * 4 pose params)
  J_regressor     (5, 5023)    joint regressor (5 joints for FLAME)
  weights         (5023, 5)    LBS skinning weights
  kintree_table   (2, 5)       joint hierarchy
  faces           (9976, 3)    triangle indices (int32)
  landmark_indices (105,)      vertex indices for 105 landmarks (or 68/51 depending on version)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _to_tensor(x, dtype=torch.float32, device="cpu"):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy()).to(dtype=dtype, device=device)
    return x.to(dtype=dtype, device=device)


def batch_rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (N, 3) to rotation matrices (N, 3, 3)."""
    angle = torch.norm(rot_vecs, dim=1, keepdim=True).clamp(min=1e-8)
    axis = rot_vecs / angle
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    K = torch.zeros(rot_vecs.shape[0], 3, 3, device=rot_vecs.device, dtype=rot_vecs.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] =  axis[:, 1]
    K[:, 1, 0] =  axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] =  axis[:, 0]
    I = torch.eye(3, device=rot_vecs.device, dtype=rot_vecs.dtype).unsqueeze(0)
    return I + sin * K + (1 - cos) * torch.bmm(K, K)


def lbs(vertices: torch.Tensor, pose: torch.Tensor,
        J: torch.Tensor, parents: torch.Tensor,
        lbs_weights: torch.Tensor, pose_dirs: torch.Tensor) -> torch.Tensor:
    """
    Linear Blend Skinning.
    vertices : (B, V, 3)
    pose     : (B, J*3)  axis-angle per joint
    J        : (B, J, 3) joint locations
    parents  : (J,)      parent indices (-1 for root)
    lbs_weights: (V, J)
    pose_dirs  : (V*3, (J-1)*9)
    Returns  : (B, V, 3)
    """
    B, V, _ = vertices.shape
    J_n = J.shape[1]

    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(B, J_n, 3, 3)

    # Pose corrective blendshapes (exclude root joint)
    pose_feature = (rot_mats[:, 1:, :, :] - torch.eye(3, device=vertices.device)).reshape(B, -1)
    pose_offsets = torch.einsum('bi,vij->bvj',
                                pose_feature,
                                pose_dirs.reshape(V, 3, -1).permute(0, 2, 1))
    # pose_dirs shape: (V*3, (J-1)*9)  → reshape to (V, 3, (J-1)*9)
    # einsum: (B, (J-1)*9) x (V, (J-1)*9, 3) → (B, V, 3)
    pd = pose_dirs.reshape(V, 3, -1)            # (V, 3, K)
    pose_offsets = torch.einsum('bk,vck->bvc', pose_feature, pd)  # (B, V, 3)

    verts_posed = vertices + pose_offsets

    # Forward kinematics
    J_transformed = []
    R_global = []
    for j in range(J_n):
        if parents[j] < 0:
            R_j = rot_mats[:, j]           # (B, 3, 3)
            t_j = J[:, j]                  # (B, 3)
        else:
            R_j = torch.bmm(R_global[parents[j]], rot_mats[:, j])
            t_j = torch.bmm(R_global[parents[j]], (J[:, j] - J[:, parents[j]]).unsqueeze(-1)).squeeze(-1) + J_transformed[parents[j]]
        J_transformed.append(t_j)
        R_global.append(R_j)

    J_transformed = torch.stack(J_transformed, dim=1)   # (B, J, 3)
    R_global = torch.stack(R_global, dim=1)              # (B, J, 3, 3)

    # Build 4x4 transformation matrices
    T = torch.zeros(B, J_n, 4, 4, device=vertices.device, dtype=vertices.dtype)
    T[:, :, :3, :3] = R_global
    T[:, :, :3, 3] = J_transformed - torch.bmm(R_global, J[:, :, :, None]).squeeze(-1)
    T[:, :, 3, 3] = 1.0

    # Blend
    W = lbs_weights.unsqueeze(0).expand(B, -1, -1)                # (B, V, J)
    T_blend = torch.einsum('bvj,bjkl->bvkl', W, T)                # (B, V, 4, 4)
    ones = torch.ones(B, V, 1, device=vertices.device, dtype=vertices.dtype)
    v_h = torch.cat([verts_posed, ones], dim=2).unsqueeze(-1)      # (B, V, 4, 1)
    v_out = torch.matmul(T_blend, v_h).squeeze(-1)[:, :, :3]      # (B, V, 3)
    return v_out


class FLAME(nn.Module):
    """
    Differentiable FLAME 2023 head model.

    Parameters
    ----------
    model_path : path to flame2023.pkl
    n_shape    : number of shape PCA components to use (default 100)
    n_expr     : number of expression PCA components to use (default 50)
    device     : 'cuda' or 'cpu'
    """

    def __init__(self, model_path: str | Path, n_shape: int = 100,
                 n_expr: int = 50, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.n_shape = n_shape
        self.n_expr = n_expr

        with open(model_path, "rb") as f:
            flame = pickle.load(f, encoding="latin1")

        # Detect layout of shapedirs / expressiondirs
        # FLAME 2023 may store them as (V, 3, K) or (V*3, K)
        def _load_basis(arr, n):
            if arr.ndim == 3:
                # (V, 3, K) → (V*3, K)
                V, _, K = arr.shape
                arr = arr.reshape(V * 3, K)
            return _to_tensor(arr[:, :n], device=device)

        v_template = _to_tensor(flame["v_template"], device=device)   # (V, 3)
        V = v_template.shape[0]

        self.register_buffer("v_template", v_template)
        self.register_buffer("shapedirs",
            _load_basis(np.array(flame["shapedirs"]), n_shape))        # (V*3, n_shape)
        self.register_buffer("expressiondirs",
            _load_basis(np.array(flame.get("expressiondirs",
                                           flame.get("expressionblendshapes",
                                                     np.zeros((V*3, 100))))), n_expr))

        # Pose corrective dirs: (V*3, K) where K = (n_joints-1)*9
        posedirs = np.array(flame.get("posedirs", np.zeros((V * 3, 36))))
        self.register_buffer("posedirs", _to_tensor(posedirs, device=device))

        # Joint regressor: (J, V)
        J_reg = np.array(flame["J_regressor"].todense()
                         if hasattr(flame["J_regressor"], "todense")
                         else flame["J_regressor"])
        self.register_buffer("J_regressor", _to_tensor(J_reg, device=device))

        # LBS weights: (V, J)
        self.register_buffer("lbs_weights",
            _to_tensor(np.array(flame["weights"]), device=device))

        # Joint hierarchy
        kintree = np.array(flame["kintree_table"])                     # (2, J)
        parents = kintree[0].astype(np.int64)
        parents[0] = -1
        self.register_buffer("parents", torch.from_numpy(parents).to(device))

        # Face triangles
        self.register_buffer("faces",
            torch.from_numpy(np.array(flame["f"] if "f" in flame else flame["faces"])
                             .astype(np.int64)).to(device))

        # Landmark indices (may be 68, 105, or 468 depending on FLAME version)
        if "landmark_indices" in flame:
            lm_idx = np.array(flame["landmark_indices"]).astype(np.int64)
        elif "lmk_faces_idx" in flame:
            # Older format: landmark barycentric coords on faces
            lm_idx = None
            self._lmk_faces_idx = torch.from_numpy(
                np.array(flame["lmk_faces_idx"])).to(device)
            self._lmk_bary_coords = _to_tensor(
                np.array(flame["lmk_bary_coords"]), device=device)
        else:
            lm_idx = None

        if lm_idx is not None:
            self.register_buffer("landmark_indices",
                torch.from_numpy(lm_idx).to(device))
        else:
            self.landmark_indices = None

        self.n_joints = self.J_regressor.shape[0]

    def forward(self, shape_params: torch.Tensor,
                expression_params: torch.Tensor,
                global_pose: torch.Tensor,
                jaw_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        shape_params      : (B, n_shape)
        expression_params : (B, n_expr)
        global_pose       : (B, 3)  axis-angle head rotation
        jaw_pose          : (B, 3)  axis-angle jaw rotation

        Returns
        -------
        vertices  : (B, V, 3)
        landmarks : (B, L, 3)  3D landmark positions
        """
        B = shape_params.shape[0]
        V = self.v_template.shape[0]

        # Shape + expression blendshapes
        shape_offset = torch.einsum("bi,ij->bj", shape_params,
                                    self.shapedirs.T).reshape(B, V, 3)
        expr_offset  = torch.einsum("bi,ij->bj", expression_params,
                                    self.expressiondirs.T).reshape(B, V, 3)
        vertices = self.v_template.unsqueeze(0) + shape_offset + expr_offset  # (B, V, 3)

        # Joint locations
        J = torch.einsum("jv,bvk->bjk", self.J_regressor, vertices)  # (B, J, 3)

        # Full pose: [global (3), jaw (3), zeros for remaining joints]
        n_extra = self.n_joints - 2
        zero_pose = torch.zeros(B, n_extra * 3, device=shape_params.device,
                                dtype=shape_params.dtype)
        pose = torch.cat([global_pose, jaw_pose, zero_pose], dim=1)   # (B, J*3)

        # LBS
        vertices = lbs(vertices, pose, J, self.parents, self.lbs_weights, self.posedirs)

        # Landmarks
        if self.landmark_indices is not None:
            landmarks = vertices[:, self.landmark_indices, :]
        elif hasattr(self, "_lmk_faces_idx"):
            # Barycentric interpolation
            f_verts = vertices[:, self.faces[self._lmk_faces_idx], :]  # (B, L, 3, 3)
            bc = self._lmk_bary_coords.unsqueeze(0).unsqueeze(-1)      # (1, L, 3, 1)
            landmarks = (f_verts * bc).sum(dim=2)                       # (B, L, 3)
        else:
            landmarks = vertices[:, :68, :]

        return vertices, landmarks
