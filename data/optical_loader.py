import numpy as np
import math
import torch
import sys
import os
from PIL import Image
from torchvision import transforms
raft_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'RAFT')
core_path = os.path.join(raft_path, 'core')
sys.path.append(core_path) 

from raft import RAFT
from utils.utils import InputPadder

def load_raft_model(model_path=None, device='cuda'):

    if model_path is None:
        model_path = os.path.join(raft_path, 'models', 'raft-kitti.pth')
    
    class Args:
        def __init__(self):
            self.model = model_path
            self.small = False
            self.mixed_precision = False
            self.alternate_corr = False
            self.dropout = 0
            self.corr_levels = 4
            self.corr_radius = 4
        
        def __contains__(self, key):
            return hasattr(self, key)
    
    args = Args()
    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.module 
    model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"✓ RAFT model loaded from: {model_path}")
    print(f"✓ Model moved to: {device}")
    
    return model

def load_image_tensor(img_tensor, device='cuda'):

    if img_tensor.max() <= 1.0:
        img_tensor = img_tensor * 255.0
    
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor.to(device)

def compute_optical_flow(raft_model, img1, img2, device='cuda'):
    
    image1 = load_image_tensor(img1, device)
    image2 = load_image_tensor(img2, device)
    
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    
    with torch.no_grad():
        flow_low, flow_up = raft_model(image1, image2, iters=20, test_mode=True)
    flow_up = padder.unpad(flow_up) 

    return flow_up
