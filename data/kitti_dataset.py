import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))


import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from data.utils_kitti import mat2euler, euler2quat
from torch.utils.data import DataLoader, SubsetRandomSampler


class KittiDataset():
    def __init__(self, root_path='../', img_height=384, img_width=1280, is_training=True, random_seed=3):
        """
        Initialize KITTI Monocular Dataset
        
        Args:
            root_path (str): Path to KITTI odometry dataset root
            img_height (int): Target image height
            img_width (int): Target image width  
            is_training (bool): Training or testing mode
            random_seed (int): Random seed for reproducibility
        """
        
        self.random_seed = random_seed
        self.img_height = img_height
        self.img_width = img_width
        self.datapath = root_path

        self.len_list = [0, 4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000, 23201] 
        self.file_map = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        
        self.T_trans = np.array([[0, 0, 1, 0],
                                [-1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]])

        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.len_list[-1] 
    
    def read_calib_file(self, path):

        data = {}
        
        with open(path, 'r') as f:
            for line in f.readlines():
                if not line.strip():
                    continue                 
                key, value = line.split(':', 1)
                value = value.strip()
                if key == 'P2':
                    P2 = np.array(list(map(float, value.split(' ')))).reshape(3, 4)
                    data['P2'] = P2
                    data['K'] = P2[:, :3]
                    break  
        
        return data
    
    def __getitem__(self, index):
        """
        Get item from dataset
        
        Args:
            index (int): Dataset index
            
        Returns:
            tuple: (img1, img2, q_gt, t_gt)
                - img1, img2: Consecutive images as tensors [C, H, W]
                - q_gt: Ground truth quaternion [4]
                - t_gt: Ground truth translation [3, 1]
        """

        for seq_idx, seq_num in enumerate(self.len_list):
            if index < seq_num:
                cur_seq = seq_idx - 1
                cur_idx_img2 = index - self.len_list[seq_idx-1]
                
                if cur_idx_img2 == 0:
                    cur_idx_img1 = 0
                else:
                    cur_idx_img1 = cur_idx_img2 - 1  
                break

        calib_path = os.path.join('data/data_odometry_calib/dataset/sequences', 
                                str(cur_seq).zfill(2), 'calib.txt')
        calib_data = self.read_calib_file(calib_path)
        K = calib_data['K']  

        cur_img_dir = os.path.join(self.datapath, self.file_map[cur_seq])
        img1_path = os.path.join(cur_img_dir, 'image_2', str(cur_idx_img1).zfill(6) + '.png')
        img2_path = os.path.join(cur_img_dir, 'image_2', str(cur_idx_img2).zfill(6) + '.png')

        # Deal with K 
        with Image.open(img1_path) as img_origin:
            orig_W, orig_H = img_origin.size
        sx = self.img_width / orig_W
        sy = self.img_height / orig_H
        K_resized = K.copy()
        K_resized[0, 0] *= sx   
        K_resized[0, 2] *= sx  
        K_resized[1, 1] *= sy   
        K_resized[1, 2] *= sy  

        img1 = self.transform(Image.open(img1_path).convert('RGB'))
        img2 = self.transform(Image.open(img2_path).convert('RGB'))
        
        pose_file = 'data/ground_truth_pose/kitti_T_diff/' + self.file_map[cur_seq] + '_diff.npy'
        pose = np.load(pose_file)
        T_diff = pose[cur_idx_img2:cur_idx_img2 + 1, :].reshape(3, 4)
        
        filler = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)
        T_gt = np.concatenate([T_diff, filler], axis=0)

        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]  # [3, 1]
        
        z_gt, y_gt, x_gt = mat2euler(R_gt)
        q_gt = euler2quat(z=z_gt, y=y_gt, x=x_gt)

        return img1, img2, q_gt, t_gt, K, K_resized
    

def get_data_loaders(data_root, img_height, img_width,
                     train_list, val_list,
                     batch_size, num_workers=4):

    train_dataset = KittiDataset(
        root_path   = data_root,
        img_height  = img_height,
        img_width   = img_width
    )
    val_dataset = KittiDataset(
        root_path   = data_root,
        img_height  = img_height,
        img_width   = img_width
    )

    boundary = train_dataset.len_list 

    train_indices = []
    for seq in train_list:
        start = boundary[seq]
        end   = boundary[seq + 1]
        train_indices += list(range(start, end))

    val_indices = []
    for seq in val_list:
        start = boundary[seq]
        end   = boundary[seq + 1]
        val_indices += list(range(start, end))

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler   = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=False
    )

    return train_indices, val_indices, train_loader, val_loader

    
