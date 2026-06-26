import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
from src.config import (
    NS_PRESET_VIEWS_ROOT, GEN_FLAME_PARAMS_PATH,
    SINGLEVIEW_PRC_ROOT, SINGLEVIEW_TRACKED_ROOT
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

class MeshUNetSVVideoTrainDataset(Dataset):
    def __init__(
            self,
            pid='416',
            res=(512, 512),
            uv_res=(512, 512),
            ds_rate=1,
            start_frame=0,
            eval_last_n_frames=600,
            num_frames=64,
    ):
        self.res = res
        self.uv_res = uv_res
        self.ds_rate = ds_rate
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

        self.pid_root = f"{SINGLEVIEW_PRC_ROOT}/{self.pid}_whiteBg_staticOffset_maskBelowLine"
        self.pid_flame_dir = f"{SINGLEVIEW_TRACKED_ROOT}/{self.pid}_whiteBg_staticOffset"
        self.img_dir = f"{self.pid_root}/images"
        self.mask_dir = f"{self.pid_root}/fg_masks"
        self.cam_dir = f"{self.pid_root}/transforms.json"
        self.pid_flametex_path = os.path.join(self.pid_flame_dir, sorted(os.listdir(self.pid_flame_dir))[-1], 'eval_30/mesh/frame_00000.jpg')
        self.pid_flame_path = os.path.join(self.pid_flame_dir, sorted(os.listdir(self.pid_flame_dir))[-1], 'tracked_flame_params_30.npz')

        if not os.path.exists(self.pid_flametex_path) or not os.path.exists(self.pid_flame_path):
            raise FileNotFoundError

        self.tex_uv = self.uv_transform(Image.open(self.pid_flametex_path).convert('RGB'))
        self.flame_params = np.load(self.pid_flame_path)

        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir) or not os.path.exists(self.cam_dir):
            raise FileNotFoundError

        self.camera_params = json.load(open(self.cam_dir))

        fids = [(fid, ts) for fid, ts in enumerate(self.flame_params['timestep_id'])][start_frame:-eval_last_n_frames]
        if len(fids) >= num_frames:
            indices = np.linspace(0, len(fids) - 1, num_frames, dtype=int)
            self.fids = [fids[_idx] for _idx in indices]
        else:
            raise ValueError("num_frames is larger than the number of frames in the video")
        self.fidxs = [fid for fid, ts in enumerate(self.fids)]
        self.total_frames = len(self.fids)

        self.enc_flame_shape = self.flame_params['shape']
        self.enc_flame_stoffset = self.flame_params['static_offset'][start_frame]

        print(f"TRAIN ID: {self.pid} // NUM_TRAIN_FRAMES: {self.total_frames}")

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, index):
        pid = self.pid

        view_id = 0
        camera_params = self.camera_params['frames'][index]
        cam_extrinsics = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        cam_intrinsic = np.array([
            [camera_params['fl_x'], 0.0, camera_params['cx']],
            [0.0, camera_params['fl_y'], camera_params['cy']],
            [0.0, 0.0, 1.0],
        ])
        cam_intrinsic = torch.from_numpy(cam_intrinsic)
        cam_extrinsic = torch.from_numpy(cam_extrinsics[:3])
        R = cam_extrinsic[:, :3].transpose(1, 0).type(torch.FloatTensor)
        R[:3, [1, 2]] *= -1
        t = cam_extrinsic[:, 3].type(torch.FloatTensor)

        flame_shape = self.flame_params['shape']                  # (300,)

        _fid = self.fids[index]
        fid, ts = _fid
        flame_rot = self.flame_params['rotation'][fid]            # (3,)
        flame_trans = self.flame_params['translation'][fid]       # (3,)
        flame_neck = self.flame_params['neck_pose'][fid]          # (3,)
        flame_jaw = self.flame_params['jaw_pose'][fid]            # (3,)
        flame_eyes = self.flame_params['eyes_pose'][fid]          # (6,)
        flame_expr = self.flame_params['expr'][fid]               # (100,)

        img_name = os.path.join(self.img_dir, f"{ts[1:]}_{view_id:02d}.png")
        mask_name = os.path.join(self.mask_dir, f"{ts[1:]}_{view_id:02d}.png")
        image = self.to_tensor(Image.open(img_name))
        mask = self.to_tensor(Image.open(mask_name))[[0]]

        sample = {
            'id': pid,
            'fids': fid,
            # 'tex_uv': self.tex_uv,
            'enc_flame_shape': self.enc_flame_shape,
            'enc_flame_stoffset': self.enc_flame_stoffset,
            'flame_rot': flame_rot,
            'flame_trans': flame_trans,
            'flame_neck': flame_neck,
            'flame_jaw': flame_jaw,
            'flame_eyes': flame_eyes,
            'flame_shape': flame_shape,
            'flame_expr': flame_expr,
            # [T, V, C, H, W] per ID
            'mv_imgs': image.view(3, self.res[0], self.res[1]),
            'mv_masks': mask.view(1, self.res[0], self.res[1]),
            'cam_K': cam_intrinsic.view(3, 3),
            'extr_R': R.view(3, 3),
            'extr_t': t,
        }

        return sample


class MeshUNetSVVideoTestDataset(Dataset):
    def __init__(self,
                pid='416',
                res=(512, 512),
                uv_res=(512, 512),
                ds_rate=1,
                train_num_frames=64,
                start_frame=0,
                eval_num_frames=600,
                ):
        self.res = res
        self.uv_res = uv_res
        self.ds_rate = ds_rate
        self.eval_num_frames = eval_num_frames
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

        self.pid_root = f"{SINGLEVIEW_PRC_ROOT}/{self.pid}_whiteBg_staticOffset_maskBelowLine"
        self.pid_flame_dir = f"{SINGLEVIEW_TRACKED_ROOT}/{self.pid}_whiteBg_staticOffset"
        self.img_dir = f"{self.pid_root}/images"
        self.mask_dir = f"{self.pid_root}/fg_masks"
        self.cam_dir = f"{self.pid_root}/transforms.json"
        self.pid_flametex_path = os.path.join(self.pid_flame_dir, sorted(os.listdir(self.pid_flame_dir))[-1], 'eval_30/mesh/frame_00000.jpg')
        self.pid_flame_path = os.path.join(self.pid_flame_dir, sorted(os.listdir(self.pid_flame_dir))[-1], 'tracked_flame_params_30.npz')

        if not os.path.exists(self.pid_flametex_path) or not os.path.exists(self.pid_flame_path):
            raise FileNotFoundError

        self.tex_uv = self.uv_transform(Image.open(self.pid_flametex_path).convert('RGB'))
        self.flame_params = np.load(self.pid_flame_path)

        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir) or not os.path.exists(self.cam_dir):
            raise FileNotFoundError

        self.camera_params = json.load(open(self.cam_dir))
        fids = [(fid, ts) for fid, ts in enumerate(self.flame_params['timestep_id'])]
        self.train_num_frames = train_num_frames
        # skip the last frame (bad tracking) and use the N frames before it
        self.fids = fids[-eval_num_frames-1:-1]
        self.fidxs = [fid for fid, ts in enumerate(self.fids)]
        self.enc_flame_shape = self.flame_params['shape']
        self.enc_flame_stoffset = self.flame_params['static_offset'][start_frame]

        print(f"TEST ID: {self.pid} // NUM_EVAL_FRAMES: {self.eval_num_frames}")

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, index):
        pid = self.pid

        view_id = 0
        camera_params = self.camera_params['frames'][index]
        cam_extrinsics = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        cam_intrinsic = np.array([
            [camera_params['fl_x'], 0.0, camera_params['cx']],
            [0.0, camera_params['fl_y'], camera_params['cy']],
            [0.0, 0.0, 1.0],
        ])
        cam_intrinsic = torch.from_numpy(cam_intrinsic)
        cam_extrinsic = torch.from_numpy(cam_extrinsics[:3])
        R = cam_extrinsic[:, :3].transpose(1, 0).type(torch.FloatTensor)
        R[:3, [1, 2]] *= -1
        t = cam_extrinsic[:, 3].type(torch.FloatTensor)

        flame_shape = self.flame_params['shape']                  # (300,)

        _fid = self.fids[index]
        fid, ts = _fid
        flame_rot = self.flame_params['rotation'][fid]            # (3,)
        flame_trans = self.flame_params['translation'][fid]       # (3,)
        flame_neck = self.flame_params['neck_pose'][fid]          # (3,)
        flame_jaw = self.flame_params['jaw_pose'][fid]            # (3,)
        flame_eyes = self.flame_params['eyes_pose'][fid]          # (6,)
        flame_expr = self.flame_params['expr'][fid]               # (100,)

        img_name = os.path.join(self.img_dir, f"{ts[1:]}_{view_id:02d}.png")
        mask_name = os.path.join(self.mask_dir, f"{ts[1:]}_{view_id:02d}.png")
        image = self.to_tensor(Image.open(img_name))
        mask = self.to_tensor(Image.open(mask_name))[[0]]

        sample = {
            'id': pid,
            'fids': fid,
            # 'tex_uv': self.tex_uv,
            'enc_flame_shape': self.enc_flame_shape,
            'enc_flame_stoffset': self.enc_flame_stoffset,
            'flame_rot': flame_rot,
            'flame_trans': flame_trans,
            'flame_neck': flame_neck,
            'flame_jaw': flame_jaw,
            'flame_eyes': flame_eyes,
            'flame_shape': flame_shape,
            'flame_expr': flame_expr,
            # [T, V, C, H, W] per ID
            'mv_imgs': image.view(3, self.res[0], self.res[1]),
            'mv_masks': mask.view(1, self.res[0], self.res[1]),
            'cam_K': cam_intrinsic.view(3, 3),
            'extr_R': R.view(3, 3),
            'extr_t': t,
        }

        return sample


def restore_to_512(padded_tensor, orig_size=512, resized_size=550, target_h=802):
    """[C, H, W] tensor: 802x550 -> center crop 550x550 -> resize 512x512."""
    if padded_tensor.ndim != 3 or padded_tensor.shape[0] not in [1, 3]:
        raise ValueError("Input must be a [C, H, W] tensor.")

    C, H, W = padded_tensor.shape
    if (H, W) != (target_h, resized_size):
        raise ValueError(f"Expected {target_h}x{resized_size}, got {H}x{W}")

    y_off = (target_h - resized_size) // 2
    cropped = padded_tensor[:, y_off:y_off+resized_size, :]

    restored = F.interpolate(
        cropped.unsqueeze(0),
        size=(orig_size, orig_size),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    return restored

class MeshUNetSVVideoRandExprTrainDataset(Dataset):
    def __init__(
            self,
            pid='416',
            res=(512, 512),
            uv_res=(512, 512),
            ds_rate=1,
            start_frame=0,
            eval_last_n_frames=600,
            num_real_frames=64,
            num_synth_frames=64,
            exp_path=None
    ):
        self.res = res
        self.uv_res = uv_res
        self.ds_rate = ds_rate
        self.to_tensor = transforms.ToTensor()
        self.uv_transform = transforms.Compose([
            transforms.Resize((uv_res[0], uv_res[1])),
            transforms.ToTensor()
        ])
        self.pid = pid

        self.pid_root = f"{SINGLEVIEW_PRC_ROOT}/{self.pid}_whiteBg_staticOffset_maskBelowLine"
        self.pid_flame_dir = f"{SINGLEVIEW_TRACKED_ROOT}/{self.pid}_whiteBg_staticOffset"
        self.img_dir = f"{self.pid_root}/images"
        self.mask_dir = f"{self.pid_root}/fg_masks"
        self.cam_dir = f"{self.pid_root}/transforms.json"
        self.pid_flametex_path = os.path.join(self.pid_flame_dir, sorted(os.listdir(self.pid_flame_dir))[-1], 'eval_30/mesh/frame_00000.jpg')
        self.pid_flame_path = os.path.join(self.pid_flame_dir, sorted(os.listdir(self.pid_flame_dir))[-1], 'tracked_flame_params_30.npz')

        if not os.path.exists(self.pid_flametex_path) or not os.path.exists(self.pid_flame_path):
            raise FileNotFoundError

        self.tex_uv = self.uv_transform(Image.open(self.pid_flametex_path).convert('RGB'))
        self.flame_params = np.load(self.pid_flame_path)

        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir) or not os.path.exists(self.cam_dir):
            raise FileNotFoundError

        self.camera_params = json.load(open(self.cam_dir))

        fids = [(fid, ts) for fid, ts in enumerate(self.flame_params['timestep_id'])][start_frame:-eval_last_n_frames]
        if len(fids) >= num_real_frames:
            indices = np.linspace(0, len(fids) - 1, num_real_frames, dtype=int)
            self.fids = [fids[_idx] for _idx in indices]
        else:
            raise ValueError("num_frames is larger than the number of frames in the video")
        self.fidxs = [fid for fid, ts in enumerate(self.fids)]
        self.num_real_frames = len(self.fids)

        self.enc_flame_shape = self.flame_params['shape']
        self.enc_flame_stoffset = self.flame_params['static_offset'][start_frame]

        # synthetic dataset paths
        self.exp_path = exp_path
        self.synth_flame_params = np.load(str(GEN_FLAME_PARAMS_PATH))
        self.synth_dataset_path = f"{self.exp_path}/rand_expr_dataset"
        self.synth_img_dir = self.synth_dataset_path
        self.synth_view_ids = np.load(f"{self.synth_dataset_path}/view_idxs.npy")[:num_synth_frames]
        self.num_synth_frames = len(self.synth_view_ids)
        self.virtual_Rs, self.virtual_ts, self.virtual_Ks = load_virtual_cams()

        print(f"TRAIN ID: {self.pid} // NUM_REAL_FRAMES: {self.num_real_frames} // NUM_SYNTH_FRAMES: {self.num_synth_frames}")

    def __len__(self):
        return len(self.fids) + self.num_synth_frames

    def __getitem__(self, index):
        pid = self.pid
        res = self.res

        if index < self.num_real_frames:
            is_real = True
            view_id = 0
            camera_params = self.camera_params['frames'][index]
            cam_extrinsics = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            cam_intrinsic = np.array([
                [camera_params['fl_x'], 0.0, camera_params['cx']],
                [0.0, camera_params['fl_y'], camera_params['cy']],
                [0.0, 0.0, 1.0],
            ])
            cam_intrinsic = torch.from_numpy(cam_intrinsic)
            cam_extrinsic = torch.from_numpy(cam_extrinsics[:3])
            R = cam_extrinsic[:, :3].transpose(1, 0).type(torch.FloatTensor)
            R[:3, [1, 2]] *= -1
            t = cam_extrinsic[:, 3].type(torch.FloatTensor)

            flame_shape = self.flame_params['shape']
            _fid = self.fids[index]
            fid, ts = _fid
            flame_rot   = self.flame_params['rotation'][fid]
            flame_trans = self.flame_params['translation'][fid]
            flame_neck  = self.flame_params['neck_pose'][fid]
            flame_jaw   = self.flame_params['jaw_pose'][fid]
            flame_eyes  = self.flame_params['eyes_pose'][fid]
            flame_expr  = self.flame_params['expr'][fid]

            img_name  = os.path.join(self.img_dir,  f"{ts[1:]}_{view_id:02d}.png")
            mask_name = os.path.join(self.mask_dir, f"{ts[1:]}_{view_id:02d}.png")
            image = self.to_tensor(Image.open(img_name))
            mask  = self.to_tensor(Image.open(mask_name))[[0]]

        else:
            is_real = False
            synth_index = index - self.num_real_frames
            view_id = self.synth_view_ids[synth_index]
            camera_params = self.camera_params['frames'][0]
            cam_intrinsic = np.array([
                [camera_params['fl_x'] * 1.5, 0.0, camera_params['cx']],
                [0.0, camera_params['fl_y'] * 1.5, camera_params['cy']],
                [0.0, 0.0, 1.0],
            ])
            cam_intrinsic = torch.from_numpy(cam_intrinsic)
            R = self.virtual_Rs[view_id]
            t = self.virtual_ts[view_id]

            flame_shape = self.synth_flame_params['shape']
            flame_rot   = self.synth_flame_params['rotation'][synth_index]
            flame_trans = self.synth_flame_params['translation'][synth_index]
            flame_neck  = self.synth_flame_params['neck_pose'][synth_index]
            flame_jaw   = self.synth_flame_params['jaw_pose'][synth_index]
            flame_eyes  = self.synth_flame_params['eyes_pose'][synth_index]
            flame_expr  = self.synth_flame_params['expr'][synth_index]

            img_name = os.path.join(self.synth_img_dir, f"{synth_index:05d}_fix.png")
            image = restore_to_512(self.to_tensor(Image.open(img_name)))
            mask  = (image < 0.95).all(dim=0, keepdim=True).float()

        sample = {
            'id': pid,
            'is_real': is_real,
            'enc_flame_shape': self.enc_flame_shape,
            'enc_flame_stoffset': self.enc_flame_stoffset,
            'flame_rot':   flame_rot,
            'flame_trans': flame_trans,
            'flame_neck':  flame_neck,
            'flame_jaw':   flame_jaw,
            'flame_eyes':  flame_eyes,
            'flame_shape': flame_shape,
            'flame_expr':  flame_expr,
            'mv_imgs':  image.view(3, res[0], res[1]),
            'mv_masks': mask.view(1, res[0], res[1]),
            'cam_K':   cam_intrinsic.view(3, 3),
            'extr_R':  R.view(3, 3),
            'extr_t':  t,
        }

        return sample
