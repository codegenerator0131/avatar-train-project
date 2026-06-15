"""
FLAME 2023 forward model.

Loads flame2023.pkl and mediapipe_landmark_embedding.npz and exposes a
differentiable forward pass:
  vertices, landmarks = FLAME(shape, expression, pose, jaw_pose)

The MediaPipe landmark embedding maps 478 MediaPipe face mesh points to
FLAME vertices via barycentric coordinates on faces — giving precise
alignment between detected 2D landmarks and the 3D model.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _load_flame_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


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
    vertices   : (B, V, 3)
    pose       : (B, J*3)  axis-angle per joint
    J          : (B, J, 3) joint locations
    parents    : (J,)      parent indices (-1 for root)
    lbs_weights: (V, J)
    pose_dirs  : (V*3, (J-1)*9)
    Returns    : (B, V, 3)
    """
    B, V, _ = vertices.shape
    J_n = J.shape[1]

    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(B, J_n, 3, 3)

    # Pose corrective blendshapes (exclude root joint)
    pose_feature = (rot_mats[:, 1:] - torch.eye(3, device=vertices.device)).reshape(B, -1)
    pd = pose_dirs.reshape(V, 3, -1)                                   # (V, 3, K)
    pose_offsets = torch.einsum('bk,vck->bvc', pose_feature, pd)       # (B, V, 3)
    verts_posed = vertices + pose_offsets

    # Forward kinematics
    J_transformed, R_global = [], []
    for j in range(J_n):
        if parents[j] < 0:
            R_j = rot_mats[:, j]
            t_j = J[:, j]
        else:
            p = parents[j].item()
            R_j = torch.bmm(R_global[p], rot_mats[:, j])
            t_j = (torch.bmm(R_global[p],
                              (J[:, j] - J[:, p]).unsqueeze(-1)).squeeze(-1)
                   + J_transformed[p])
        J_transformed.append(t_j)
        R_global.append(R_j)

    J_transformed = torch.stack(J_transformed, dim=1)   # (B, J, 3)
    R_global      = torch.stack(R_global,      dim=1)   # (B, J, 3, 3)

    # 4x4 world transforms
    T = torch.zeros(B, J_n, 4, 4, device=vertices.device, dtype=vertices.dtype)
    T[:, :, :3, :3] = R_global
    T[:, :, :3,  3] = J_transformed - torch.bmm(
        R_global.reshape(B * J_n, 3, 3),
        J.reshape(B * J_n, 3, 1)
    ).reshape(B, J_n, 3)
    T[:, :,  3,  3] = 1.0

    # Blend
    W       = lbs_weights.unsqueeze(0).expand(B, -1, -1)           # (B, V, J)
    T_blend = torch.einsum('bvj,bjkl->bvkl', W, T)                 # (B, V, 4, 4)
    ones    = torch.ones(B, V, 1, device=vertices.device, dtype=vertices.dtype)
    v_h     = torch.cat([verts_posed, ones], dim=2).unsqueeze(-1)   # (B, V, 4, 1)
    return torch.matmul(T_blend, v_h).squeeze(-1)[:, :, :3]        # (B, V, 3)


class FLAME(nn.Module):
    """
    Differentiable FLAME 2023 head model with MediaPipe landmark support.

    Parameters
    ----------
    model_path   : path to flame2023.pkl
    lm_embed_path: path to mediapipe_landmark_embedding.npz
    n_shape      : number of shape PCA components (default 100)
    n_expr       : number of expression PCA components (default 50)
    device       : 'cuda' or 'cpu'
    """

    def __init__(self, model_path: str | Path,
                 lm_embed_path: str | Path | None = None,
                 n_shape: int = 100, n_expr: int = 50,
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        self.n_shape = n_shape
        self.n_expr  = n_expr

        # ---- Load FLAME pkl ----------------------------------------------
        flame = _load_flame_pkl(model_path)

        def _load_basis(arr, n):
            arr = np.array(arr)
            if arr.ndim == 3:                      # (V, 3, K) → (V*3, K)
                arr = arr.reshape(arr.shape[0] * 3, arr.shape[2])
            return _to_tensor(arr[:, :n], device=device)

        v_template = _to_tensor(np.array(flame["v_template"]), device=device)
        V = v_template.shape[0]

        self.register_buffer("v_template", v_template)
        self.register_buffer("shapedirs",
            _load_basis(flame["shapedirs"], n_shape))
        self.register_buffer("expressiondirs",
            _load_basis(flame.get("expressiondirs",
                        flame.get("expressionblendshapes",
                        np.zeros((V * 3, 100)))), n_expr))

        posedirs = np.array(flame.get("posedirs", np.zeros((V * 3, 36))))
        self.register_buffer("posedirs", _to_tensor(posedirs, device=device))

        J_reg = flame["J_regressor"]
        if hasattr(J_reg, "todense"):
            J_reg = np.array(J_reg.todense())
        else:
            J_reg = np.array(J_reg)
        self.register_buffer("J_regressor", _to_tensor(J_reg, device=device))

        self.register_buffer("lbs_weights",
            _to_tensor(np.array(flame["weights"]), device=device))

        kintree  = np.array(flame["kintree_table"])
        parents  = kintree[0].astype(np.int64)
        parents[0] = -1
        self.register_buffer("parents",
            torch.from_numpy(parents).to(device))

        faces_key = "f" if "f" in flame else "faces"
        self.register_buffer("faces",
            torch.from_numpy(np.array(flame[faces_key]).astype(np.int64)).to(device))

        self.n_joints = self.J_regressor.shape[0]

        # ---- MediaPipe landmark embedding --------------------------------
        # mediapipe_landmark_embedding.npz contains:
        #   lmk_face_idx   (478,)    which face each landmark sits on
        #   lmk_b_coords   (478, 3)  barycentric coords on that face
        if lm_embed_path is not None and Path(lm_embed_path).exists():
            emb = np.load(lm_embed_path, allow_pickle=True)
            # Try common key names
            face_idx  = emb.get("lmk_face_idx",
                        emb.get("lmk_faces_idx",
                        emb.get("face_idx", None)))
            bary      = emb.get("lmk_b_coords",
                        emb.get("lmk_bary_coords",
                        emb.get("bary_coords", None)))

            if face_idx is not None and bary is not None:
                self.register_buffer("lmk_face_idx",
                    torch.from_numpy(np.array(face_idx).astype(np.int64)).to(device))
                self.register_buffer("lmk_bary_coords",
                    _to_tensor(np.array(bary), device=device))
                self.n_landmarks = int(self.lmk_face_idx.shape[0])
                # Try to load the MediaPipe index map (which of 478 MP points each row corresponds to)
                mp_idx = emb.get("lmk_mp_idx",
                          emb.get("mp_idx",
                          emb.get("mediapipe_idx", None)))
                self.lmk_mp_idx = np.array(mp_idx).astype(np.int64) if mp_idx is not None else None
                print(f"  Loaded MediaPipe embedding: {self.n_landmarks} landmarks "
                      f"| keys: {list(emb.keys())}"
                      f"{' | mp_idx: YES' if self.lmk_mp_idx is not None else ' | mp_idx: NOT FOUND (will use sequential)'}")
            else:
                print(f"  WARNING: could not read embedding keys from {lm_embed_path}")
                print(f"  Available keys: {list(emb.keys())}")
                self._init_fallback_landmarks(flame, V, device)
        else:
            self._init_fallback_landmarks(flame, V, device)

    def _init_fallback_landmarks(self, flame: dict, V: int, device: str) -> None:
        """Fall back to vertex-index landmarks if embedding not available."""
        self.lmk_face_idx   = None
        self.lmk_bary_coords = None
        if "landmark_indices" in flame:
            idx = np.array(flame["landmark_indices"]).astype(np.int64)
            self.register_buffer("landmark_indices",
                torch.from_numpy(idx).to(device))
            self.n_landmarks = int(idx.shape[0])
            print(f"  Using vertex-index landmarks: {self.n_landmarks} landmarks")
        elif "lmk_faces_idx" in flame:
            self.register_buffer("lmk_face_idx",
                torch.from_numpy(np.array(flame["lmk_faces_idx"]).astype(np.int64)).to(device))
            self.register_buffer("lmk_bary_coords",
                _to_tensor(np.array(flame["lmk_bary_coords"]), device=device))
            self.n_landmarks = int(self.lmk_face_idx.shape[0])
            print(f"  Using built-in barycentric landmarks: {self.n_landmarks} landmarks")
        else:
            self.n_landmarks = 68
            print("  WARNING: no landmark info found, using first 68 vertices")

    def get_landmarks(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Extract landmark positions from a posed mesh.
        vertices : (B, V, 3)
        Returns  : (B, L, 3)
        """
        if hasattr(self, "lmk_face_idx") and self.lmk_face_idx is not None:
            # Barycentric interpolation: sample points on face triangles
            B = vertices.shape[0]
            face_verts = vertices[:, self.faces[self.lmk_face_idx], :]  # (B, L, 3, 3)
            bc = self.lmk_bary_coords.unsqueeze(0).unsqueeze(-1)        # (1, L, 3, 1)
            return (face_verts * bc).sum(dim=2)                          # (B, L, 3)
        elif hasattr(self, "landmark_indices"):
            return vertices[:, self.landmark_indices, :]
        else:
            return vertices[:, :self.n_landmarks, :]

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
        global_pose       : (B, 3)  axis-angle global head rotation
        jaw_pose          : (B, 3)  axis-angle jaw rotation

        Returns
        -------
        vertices  : (B, V, 3)
        landmarks : (B, L, 3)
        """
        B = shape_params.shape[0]
        V = self.v_template.shape[0]

        shape_offset = torch.einsum("bi,ij->bj",
                                    shape_params, self.shapedirs.T).reshape(B, V, 3)
        expr_offset  = torch.einsum("bi,ij->bj",
                                    expression_params, self.expressiondirs.T).reshape(B, V, 3)
        vertices = self.v_template.unsqueeze(0) + shape_offset + expr_offset

        J = torch.einsum("jv,bvk->bjk", self.J_regressor, vertices)

        n_extra   = self.n_joints - 2
        zero_pose = torch.zeros(B, n_extra * 3,
                                device=shape_params.device, dtype=shape_params.dtype)
        pose = torch.cat([global_pose, jaw_pose, zero_pose], dim=1)

        vertices  = lbs(vertices, pose, J, self.parents, self.lbs_weights, self.posedirs)
        landmarks = self.get_landmarks(vertices)

        return vertices, landmarks
