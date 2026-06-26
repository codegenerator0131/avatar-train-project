import os
import json
import numpy as np
import random
import math
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
from src.config import (
    NS_PRESET_VIEWS_ROOT, GEN_FLAME_PARAMS_PATH,
    SINGLEVIEW_PRC_ROOT, SINGLEVIEW_TRACKED_ROOT,
)

NERSEMBLE_VIEWS = {
    0: 'cam_222200042',
    1: 'cam_222200044',
    2: 'cam_222200046',
    3: 'cam_222200040',
    4: 'cam_222200036',
    5: 'cam_222200048',
    6: 'cam_220700191',
    7: 'cam_222200041',
    8: 'cam_222200037',
    9: 'cam_222200038',
    10: 'cam_222200047',
    11: 'cam_222200043',
    12: 'cam_222200049',
    13: 'cam_222200039',
    14: 'cam_222200045',
    15: 'cam_221501007'
}

def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """R: [3,3], return quat as [w,x,y,z] (normalized)."""
    t = R.trace()
    if t > 0:
        r = torch.sqrt(1.0 + t)
        w = 0.5 * r
        r2 = 0.5 / r
        x = (R[2,1] - R[1,2]) * r2
        y = (R[0,2] - R[2,0]) * r2
        z = (R[1,0] - R[0,1]) * r2
    else:
        i = torch.argmax(torch.tensor([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            r = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            x = 0.5 * r
            r2 = 0.5 / r
            y = (R[0,1] + R[1,0]) * r2
            z = (R[0,2] + R[2,0]) * r2
            w = (R[2,1] - R[1,2]) * r2
        elif i == 1:
            r = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            y = 0.5 * r
            r2 = 0.5 / r
            x = (R[0,1] + R[1,0]) * r2
            z = (R[1,2] + R[2,1]) * r2
            w = (R[0,2] - R[2,0]) * r2
        else:
            r = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            z = 0.5 * r
            r2 = 0.5 / r
            x = (R[0,2] + R[2,0]) * r2
            y = (R[1,2] + R[2,1]) * r2
            w = (R[1,0] - R[0,1]) * r2
    q = torch.stack([w, x, y, z], dim=0)
    q = q / (q.norm() + 1e-8)
    return q

def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """q: [w,x,y,z] normalized -> R[3,3]."""
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    R = torch.empty(3,3, dtype=torch.float32, device=q.device)
    R[0,0] = ww + xx - yy - zz
    R[0,1] = 2*(x*y - w*z)
    R[0,2] = 2*(x*z + w*y)
    R[1,0] = 2*(x*y + w*z)
    R[1,1] = ww - xx + yy - zz
    R[1,2] = 2*(y*z - w*x)
    R[2,0] = 2*(x*z - w*y)
    R[2,1] = 2*(y*z + w*x)
    R[2,2] = ww - xx - yy + zz
    return R

def _slerp(q0: torch.Tensor, q1: torch.Tensor, u: float) -> torch.Tensor:
    """Spherical linear interpolation between quaternions [w,x,y,z]."""
    q0 = q0 / (q0.norm() + 1e-8)
    q1 = q1 / (q1.norm() + 1e-8)
    dot = torch.dot(q0, q1)
    if dot < 0.0:  # ensure shortest arc
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly parallel: lerp and renormalize
        q = (1.0 - u) * q0 + u * q1
        return q / (q.norm() + 1e-8)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * u
    s0 = torch.sin(theta_0 - theta) / (sin_theta_0 + 1e-8)
    s1 = torch.sin(theta) / (sin_theta_0 + 1e-8)
    return s0 * q0 + s1 * q1


def load_preset_cam():
    all_intrinsics = []
    all_extr_R = []
    all_extr_t = []
    cam_intrinsic = np.load(str(NS_PRESET_VIEWS_ROOT / 'nersemble_intrinsic.npy'))   # [3, 3]
    cam_intrinsic[:2] *= 1 / 4
    cam_intrinsic = torch.from_numpy(cam_intrinsic)
    preset_cam_ids = ['cam_222200049', 'cam_222200046']
    cam_extr_path_template = str(NS_PRESET_VIEWS_ROOT / 'ns_**cam_id**_extr.npy')
    for cam_id in preset_cam_ids:
        cam_extr_path = cam_extr_path_template.replace('**cam_id**', cam_id)
        cam_extrinsic = torch.from_numpy(np.load(cam_extr_path))  # [3, 4]
        R = cam_extrinsic[:, :3].transpose(1, 0).type(torch.FloatTensor)
        t = cam_extrinsic[:, 3].type(torch.FloatTensor)
        all_extr_R.append(R)
        all_extr_t.append(t)
        all_intrinsics.append(cam_intrinsic)

    return all_extr_R, all_extr_t, all_intrinsics


def load_virtual_cams():
    all_intrinsics = []
    all_extr_R = []
    all_extr_t = []
    cam_intrinsic = np.load(str(NS_PRESET_VIEWS_ROOT / 'nersemble_intrinsic.npy'))   # [3, 3]
    cam_intrinsic[:2] *= 1 / 4
    cam_intrinsic = torch.from_numpy(cam_intrinsic)
    preset_cam_ids = [v for (k, v) in NERSEMBLE_VIEWS.items()]
    cam_extr_path_template = str(NS_PRESET_VIEWS_ROOT / 'ns_**cam_id**_extr.npy')
    for cam_id in preset_cam_ids:
        cam_extr_path = cam_extr_path_template.replace('**cam_id**', cam_id)
        cam_extrinsic = torch.from_numpy(np.load(cam_extr_path))  # [3, 4]
        R = cam_extrinsic[:, :3].transpose(1, 0).type(torch.FloatTensor)
        t = cam_extrinsic[:, 3].type(torch.FloatTensor)
        all_extr_R.append(R)
        all_extr_t.append(t)
        all_intrinsics.append(cam_intrinsic)

    return all_extr_R, all_extr_t, all_intrinsics


class MeshUNetRandomExprRenderDataset(Dataset):
    def __init__(
            self,
            pid='240',
            num_frames=None,
            res=(512, 512),
            uv_res=(512, 512),
    ):

        self.res = res
        self.uv_res = uv_res
        self.to_tensor = transforms.ToTensor()
        self.ds_transform = transforms.Compose([
            transforms.Resize((res[0], res[1])),
            transforms.ToTensor()
        ])
        self.uv_transform = transforms.Compose([
            transforms.Resize((uv_res[0], uv_res[1])),
            transforms.ToTensor()
        ])
        self.pid = pid

        self.pid_flame_dir = f"{SINGLEVIEW_TRACKED_ROOT}/{self.pid}_whiteBg_staticOffset"
        self.cam_dir = f"{SINGLEVIEW_PRC_ROOT}/{self.pid}_whiteBg_staticOffset_maskBelowLine/transforms.json"
        self.pid_flametex_path = os.path.join(self.pid_flame_dir, sorted(os.listdir(self.pid_flame_dir))[-1], 'eval_30/mesh/frame_00000.jpg')
        self.pid_flame_path = os.path.join(self.pid_flame_dir, sorted(os.listdir(self.pid_flame_dir))[-1], 'tracked_flame_params_30.npz')

        if not os.path.exists(self.pid_flametex_path) or not os.path.exists(self.pid_flame_path):
            raise FileNotFoundError

        self.pid_tex_uv = self.uv_transform(Image.open(self.pid_flametex_path).convert('RGB'))
        self.pid_flame_params = np.load(self.pid_flame_path)

        # motion data
        self.val_views = [0]
        self.sid = ""

        self.flame_params = np.load(str(GEN_FLAME_PARAMS_PATH))
        #
        self.fids = [fid for fid in range(len(self.flame_params['expr']))]  # 1000
        if num_frames is not None:
            self.fids = self.fids[:num_frames]
        self.fidxs = [fid for fid, ts in enumerate(self.fids)]
        self.camera_params = json.load(open(self.cam_dir))

        self.virtual_Rs, self.virtual_ts, self.virtual_Ks = load_virtual_cams()

        self.num_views = len(self.virtual_Rs)  # 16

        print(f"TEST ID: {self.pid} // Random Expr")

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, index):

        all_images = []
        all_masks = []
        all_intrinsics = []
        all_extr_R = []
        all_extr_t = []
        # load camera params

        fid = self.fids[index]
        flame_shape = self.flame_params['shape']
        flame_rot = self.flame_params['rotation'][fid]            # (3,)
        flame_trans = self.flame_params['translation'][fid]       # (3,)
        flame_neck = self.flame_params['neck_pose'][fid]          # (3,)
        flame_jaw = self.flame_params['jaw_pose'][fid]            # (3,)
        flame_eyes = self.flame_params['eyes_pose'][fid]          # (6,)
        flame_expr = self.flame_params['expr'][fid]               # (100,)

        pid_flame_shape = self.pid_flame_params['shape']
        pid_flame_rot = self.pid_flame_params['rotation'][0]
        pid_flame_trans = self.pid_flame_params['translation'][0]
        pid_flame_neck = self.pid_flame_params['neck_pose'][0]
        pid_flame_jaw = self.pid_flame_params['jaw_pose'][0]
        pid_flame_eyes = self.pid_flame_params['eyes_pose'][0]
        pid_flame_expr = self.pid_flame_params['expr'][0]
        pid_flame_stoffset = self.pid_flame_params['static_offset'][0]

        total_frames = len(self.fids)

        view_idx = random.randint(0, self.num_views - 1)
        all_extr_R.append(self.virtual_Rs[view_idx])
        all_extr_t.append(self.virtual_ts[view_idx])
        camera_params = self.camera_params['frames'][0]
        cam_intrinsic = np.array([
            [camera_params['fl_x'] * 1.5, 0.0, camera_params['cx']],
            [0.0, camera_params['fl_y'] * 1.5, camera_params['cy']],
            [0.0, 0.0, 1.0],
        ])
        all_intrinsics.append(torch.from_numpy(cam_intrinsic))
        all_images.append(torch.zeros(3, self.res[0], self.res[1], dtype=torch.float32))
        all_masks.append(torch.ones(1, self.res[0], self.res[1], dtype=torch.float32))


        sample = {
            'fids': fid,
            'view_idx': view_idx,
            'pid_flame_rot': pid_flame_rot,
            'pid_flame_trans': pid_flame_trans,
            'pid_flame_neck': pid_flame_neck,
            'pid_flame_jaw': pid_flame_jaw,
            'pid_flame_eyes': pid_flame_eyes,
            'pid_flame_shape': pid_flame_shape,
            'pid_flame_expr': pid_flame_expr,
            'pid_flame_stoffset': pid_flame_stoffset,
            'flame_rot': flame_rot,
            'flame_trans': flame_trans,
            'flame_neck': flame_neck,
            'flame_jaw': flame_jaw,
            'flame_eyes': flame_eyes,
            'flame_shape': flame_shape,
            'flame_expr': flame_expr,
            # [T, V, C, H, W] per ID
            'mv_imgs': torch.stack(all_images).view(len(all_images), 3, self.res[0], self.res[1]),
            'mv_masks': torch.stack(all_masks).view(len(all_masks), 1, self.res[0], self.res[1]),
            'cam_K': torch.stack(all_intrinsics).view(len(all_intrinsics), 3, 3),
            'extr_R': torch.stack(all_extr_R).view(len(all_extr_R), 3, 3),
            'extr_t': torch.stack(all_extr_t).view(len(all_extr_t), 3),
        }

        return sample
    

class TestMotionRenderDataset(Dataset):
    """Drive a personalized singleview avatar with a motion_samples directory.

    motion_dir must contain:
        images/                       -- motion frames
        fg_masks/                     -- foreground masks
        flame_param/{idx:05d}.npz    -- per-frame FLAME params
        transforms.json               -- multi-cam entries with c2w transform_matrix (NeRF/OpenGL)

    pid always uses the singleview tracked root (SINGLEVIEW_TRACKED_ROOT).
    """

    def __init__(
            self,
            pid,
            motion_dir,
            num_frames=None,
            cam_ids=None,
            res=(802, 550),
            uv_res=(512, 512),
            cam_path='motion_id',
            spherical_cycles=1,
    ):
        from pathlib import Path
        from collections import defaultdict

        self.res = res
        self.uv_res = uv_res
        self.to_tensor = transforms.ToTensor()
        self.uv_transform = transforms.Compose([
            transforms.Resize((uv_res[0], uv_res[1])),
            transforms.ToTensor(),
        ])
        self.pid = pid
        self.cam_path = cam_path
        self.cam_ids = cam_ids if cam_ids is not None else ['our_cam']

        # ── PID side (singleview, always) ─────────────────────────────────────
        pid_flame_dir = f"{SINGLEVIEW_TRACKED_ROOT}/{pid}_whiteBg_staticOffset"
        assert os.path.isdir(pid_flame_dir), f"pid flame dir not found: {pid_flame_dir}"
        last_run = sorted(os.listdir(pid_flame_dir))[-1]
        pid_flametex_path = os.path.join(pid_flame_dir, last_run, 'eval_30/mesh/frame_00000.jpg')
        pid_flame_path    = os.path.join(pid_flame_dir, last_run, 'tracked_flame_params_30.npz')
        assert os.path.isfile(pid_flametex_path), f"pid tex not found: {pid_flametex_path}"
        assert os.path.isfile(pid_flame_path),    f"pid flame not found: {pid_flame_path}"

        self.pid_tex_uv      = self.uv_transform(Image.open(pid_flametex_path).convert('RGB'))
        self.pid_flame_params = np.load(pid_flame_path)

        # ── Motion side ───────────────────────────────────────────────────────
        motion_dir = Path(motion_dir)
        assert motion_dir.is_dir(), f"motion_dir not found: {motion_dir}"
        self.motion_dir = motion_dir

        cam_file = motion_dir / 'transforms.json'
        assert cam_file.is_file(), f"transforms.json not found: {cam_file}"

        all_entries = json.load(open(str(cam_file)))['frames']
        selected = [e for e in all_entries if e['camera_id'] in self.cam_ids]
        selected.sort(key=lambda e: (e['timestep_index'], self.cam_ids.index(e['camera_id'])))

        if num_frames is not None:
            seen, kept = set(), []
            for e in selected:
                ti = e['timestep_index']
                if ti not in seen:
                    seen.add(ti)
                    if len(seen) > num_frames:
                        break
                kept.append(e)
            selected = kept

        ts_map = defaultdict(list)
        for e in selected:
            ts_map[e['timestep_index']].append(e)
        self.timesteps = sorted(ts_map.keys())
        self.ts_map    = ts_map

        # ── Preset cameras for spherical mode ─────────────────────────────────
        preset_Rs, preset_ts, preset_Ks = load_preset_cam()
        self.preset_Rs         = [R.clone().float() for R in preset_Rs]
        self.preset_ts         = [t.clone().float() for t in preset_ts]
        self.preset_Ks         = [K.clone().float() for K in preset_Ks]
        self.num_cam_keyframes = len(self.preset_Rs)
        assert self.num_cam_keyframes >= 2
        self.spherical_cycles  = int(spherical_cycles)
        self.anchor_t          = self.preset_ts[0].clone()
        self.anchor_K          = self.preset_Ks[0].clone()

        print(f"TestMotionRenderDataset | PID: {pid} | motion: {motion_dir.name} | "
              f"cams: {self.cam_ids} | timesteps: {len(self.timesteps)}")

    def __len__(self):
        return len(self.timesteps)

    @staticmethod
    def _c2w_gl_to_extr(c2w_gl: np.ndarray):
        """NeRF OpenGL c2w [4,4] → (R_c2w_opencv [3,3], t_w2c_opencv [3])."""
        c2w_cv = c2w_gl.copy()
        c2w_cv[:3, 1:3] *= -1          # OpenGL → OpenCV: flip Y, Z axes
        R_c2w  = c2w_cv[:3, :3]
        t_w2c  = -(R_c2w.T @ c2w_cv[:3, 3])
        return R_c2w, t_w2c

    def __getitem__(self, index):
        ts_idx  = self.timesteps[index]
        entries = self.ts_map[ts_idx]

        # ── PID FLAME params ──────────────────────────────────────────────────
        pp = self.pid_flame_params
        pid_flame_shape    = pp['shape']
        pid_flame_rot      = pp['rotation'][0]
        pid_flame_trans    = pp['translation'][0]
        pid_flame_neck     = pp['neck_pose'][0]
        pid_flame_jaw      = pp['jaw_pose'][0]
        pid_flame_eyes     = pp['eyes_pose'][0]
        pid_flame_expr     = pp['expr'][0]
        pid_flame_stoffset = pp['static_offset'][0]

        # ── Motion FLAME params (per-frame npz) ───────────────────────────────
        fp = np.load(str(self.motion_dir / entries[0]['flame_param_path']))
        flame_shape = fp['shape'][0]
        flame_rot   = fp['rotation'][0]
        flame_trans = fp['translation'][0]
        flame_neck  = fp['neck_pose'][0]
        flame_jaw   = fp['jaw_pose'][0]
        flame_eyes  = fp['eyes_pose'][0]
        flame_expr  = fp['expr'][0]

        all_images, all_masks, all_intrinsics, all_extr_R, all_extr_t = [], [], [], [], []

        if self.cam_path == 'spherical':
            total    = len(self.timesteps)
            progress = (index % total) / float(total)
            M        = self.num_cam_keyframes
            period   = 2 * (M - 1)
            cycles   = max(1, self.spherical_cycles)
            s        = progress * cycles * period
            s_mod    = s % period
            seg      = int(math.floor(s_mod))
            u        = float(s_mod - seg)
            if seg < M - 1:
                i0, i1 = seg, seg + 1
            else:
                j = seg - (M - 1)
                i0, i1 = (M - 1) - j, (M - 1) - j - 1
            q0 = _rotmat_to_quat(self.preset_Rs[i0])
            q1 = _rotmat_to_quat(self.preset_Rs[i1])
            R_interp = _quat_to_rotmat(_slerp(q0, q1, u))
            all_extr_R.append(R_interp)
            all_extr_t.append(self.anchor_t)
            all_intrinsics.append(self.anchor_K)
            all_images.append(torch.zeros(3, self.res[0], self.res[1]))
            all_masks.append(torch.ones(1, self.res[0], self.res[1]))

        else:  # motion_id
            for entry in entries:
                R_c2w, t_w2c = self._c2w_gl_to_extr(np.array(entry['transform_matrix'], dtype=np.float64))
                cam_K = torch.from_numpy(np.array([
                    [entry['fl_x'], 0.0,           entry['cx']],
                    [0.0,           entry['fl_y'],  entry['cy']],
                    [0.0,           0.0,            1.0],
                ], dtype=np.float64))
                img_path  = self.motion_dir / entry['file_path']
                mask_path = self.motion_dir / entry['fg_mask_path']
                assert img_path.is_file(),  f"image not found: {img_path}"
                assert mask_path.is_file(), f"mask not found: {mask_path}"
                all_images.append(self.to_tensor(Image.open(str(img_path))))
                all_masks.append(self.to_tensor(Image.open(str(mask_path)))[[0]])
                all_extr_R.append(torch.from_numpy(R_c2w).float())
                all_extr_t.append(torch.from_numpy(t_w2c).float())
                all_intrinsics.append(cam_K)

        V = len(all_images)
        return {
            'fids':               ts_idx,
            'pid_tex_uv':         self.pid_tex_uv,
            'pid_flame_rot':      pid_flame_rot,
            'pid_flame_trans':    pid_flame_trans,
            'pid_flame_neck':     pid_flame_neck,
            'pid_flame_jaw':      pid_flame_jaw,
            'pid_flame_eyes':     pid_flame_eyes,
            'pid_flame_shape':    pid_flame_shape,
            'pid_flame_expr':     pid_flame_expr,
            'pid_flame_stoffset': pid_flame_stoffset,
            'flame_rot':          flame_rot,
            'flame_trans':        flame_trans,
            'flame_neck':         flame_neck,
            'flame_jaw':          flame_jaw,
            'flame_eyes':         flame_eyes,
            'flame_shape':        flame_shape,
            'flame_expr':         flame_expr,
            'mv_imgs':  torch.stack(all_images).view(V, 3, self.res[0], self.res[1]),
            'mv_masks': torch.stack(all_masks).view(V, 1, self.res[0], self.res[1]),
            'cam_K':    torch.stack(all_intrinsics).view(V, 3, 3),
            'extr_R':   torch.stack(all_extr_R).view(V, 3, 3),
            'extr_t':   torch.stack(all_extr_t).view(V, 3),
        }
