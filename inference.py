import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))

import torch
import numpy as np

from model.FGVONet import FGVONet
from data.kitti_dataset import KittiDataset

class FGVONet_inference:
    def __init__(self,
                 checkpoint_path,
                 data_root,
                 result_dir,
                 test_list,
                 img_height=192,
                 img_width=640,
                 device='cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # load model
        self.model = FGVONet(
            img_backbone='hrnet_w32',
            img_backbone_pretrained=False
        ).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        self.test_list = test_list
        self.dataset = KittiDataset(
            root_path=data_root,
            img_height=img_height,
            img_width=img_width
        )
        self.boundary = self.dataset.len_list  # sequence boundaries
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir

    def quat2mat(self, q:
                   np.ndarray) -> np.ndarray:
        ''' Calculate rotation matrix corresponding to quaternion
        https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
        '''
        w, x, y, z = q
        Nq = w*w + x*x + y*y + z*z
        if Nq < 1e-8:
            return np.eye(3)
        s = 2.0 / Nq
        X = x*s; Y = y*s; Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z
        return np.array([
            [1.0-(yY+zZ), xY-wZ,      xZ+wY],
            [xY+wZ,       1.0-(xX+zZ), yZ-wX],
            [xZ-wY,       yZ+wX,      1.0-(xX+yY)]
        ])

    def infer_sequence(self, seq_idx: int):
        start = self.boundary[seq_idx]
        end   = self.boundary[seq_idx+1]
        T_final = np.eye(4)
        traj = []  

        for idx in range(start, end):

            img1, img2, _, _, _, _ = self.dataset[idx]

            img1 = img1.unsqueeze(0).to(self.device)
            img2 = img2.unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(img1, img2)
                q_pred = out[0].cpu().numpy().reshape(4,)
                t_pred = out[1].cpu().numpy().reshape(3,)

            R = self.quat2mat(q_pred)
            TT = np.eye(4)
            TT[:3,:3] = R
            TT[:3,3]  = t_pred

            T_final = T_final @ TT

            T3x4 = T_final[:3, :]
            traj.append(T3x4.reshape(-1))

        traj = np.stack(traj, axis=0)  # [N_frames, 12]
        out_file = os.path.join(self.result_dir, f"{seq_idx:02d}_pred.txt")
        np.savetxt(out_file, traj, fmt='%.08f')
        print(f"Saved trajectory for sequence {seq_idx} to {out_file}")

    def run(self):
        for seq in self.test_list:
            self.infer_sequence(seq)


if __name__ == '__main__':
    inference = FGVONet_inference(
        checkpoint_path='',
        data_root='',
        result_dir='',
        test_list=[7,8,9,10],
        img_height=192,
        img_width=640,
        device='cuda'
    )
    inference.run()
