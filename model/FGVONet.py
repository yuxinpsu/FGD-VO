import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))

import torch 
import torch.nn as nn
import torch.nn.functional as F
from model.img_backbone import PyramidBackbone
from data.optical_loader import load_raft_model, compute_optical_flow
from model.utils_FGVONet import FlowGuidedAttention, quat_inv, quat_mul

class FGVONet(nn.Module):
    def __init__(self, img_size=(192,640), img_backbone='hrnet_w32', img_backbone_pretrained=True, mlp1_local=(64, 64), dim_layer0=64, dim_layer1=128, 
                 dim_layer2=256, dim_layer3=512, mlp_layer3=(128,64), mlp_layer2=(128, 64), mlp_layer1=(128, 64), mlp_layer0=(128, 64), layer3_up=256, 
                 layer2_up=256, layer1_up=256, layer0_up=256, layer_drop=0.25):
        super().__init__()

        self.img_backbone = PyramidBackbone(HRNet_Name=img_backbone, HRNet_Pretrained=img_backbone_pretrained)
        self.FGA_layer3 = FlowGuidedAttention(C_in=dim_layer3, mlp1_local=mlp1_local)
        self.FGA_layer2 = FlowGuidedAttention(C_in=dim_layer2, mlp1_local=mlp1_local)
        self.FGA_layer1 = FlowGuidedAttention(C_in=dim_layer1, mlp1_local=mlp1_local)
        self.FGA_layer0 = FlowGuidedAttention(C_in=dim_layer0, mlp1_local=mlp1_local)
        self.raft = load_raft_model()


        # MLP_layer for layer3 
        in_ch_l3 = dim_layer3 + mlp1_local[-1]
        layers3 = []
        for idx, ch in enumerate(mlp_layer3):            
            layers3 += [
                nn.Conv2d(in_ch_l3 if idx == 0 else mlp_layer3[0], ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ]
        self.mlp_layer3 = nn.Sequential(*layers3)
        self.l3_up_conv = nn.Conv1d(mlp_layer3[-1], layer3_up, kernel_size=1, bias=False)  
        self.dropout_q_layer3 = nn.Dropout(p=layer_drop)
        self.dropout_t_layer3 = nn.Dropout(p=layer_drop)
        self.conv_q_layer3 = nn.Conv1d(layer3_up, 4, kernel_size=1, bias=False)
        self.conv_t_layer3 = nn.Conv1d(layer3_up, 3, kernel_size=1, bias=False)
        self.pos_embed_layer3 = nn.Parameter(torch.zeros(1, 1, (img_size[0]//16)*(img_size[1]//16)))
        nn.init.trunc_normal_(self.pos_embed_layer3, std=0.02)

        # MLP_layer for layer2
        self.img_size_layer2 = tuple(s // 8 for s in img_size)
        self.l3_weight_up_layer2 = nn.Conv2d(mlp_layer3[-1]+dim_layer2, 256, kernel_size=1, bias=False)  
        self.l3_feat_up_layer2 = nn.Conv2d(mlp1_local[-1]+dim_layer2, 256, kernel_size=1, bias=False)  

        in_ch_l2_feat = dim_layer2 + mlp1_local[-1] + 256
        layers2_feat = []
        for idx, ch in enumerate(mlp_layer2):            
            layers2_feat += [
                nn.Conv2d(in_ch_l2_feat if idx == 0 else mlp_layer2[0], ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ]
        self.mlp_layer2_feat = nn.Sequential(*layers2_feat)

        in_ch_l2_weight = dim_layer2 + mlp_layer2[-1] + 256
        layers2_weight = []
        for idx, ch in enumerate(mlp_layer2):            
            layers2_weight += [
                nn.Conv2d(in_ch_l2_weight if idx == 0 else mlp_layer2[0], ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ]
        self.mlp_layer2_weight = nn.Sequential(*layers2_weight)

        self.l2_up_conv = nn.Conv1d(mlp_layer2[-1], layer2_up, kernel_size=1, bias=False) 
        self.dropout_q_layer2 = nn.Dropout(p=layer_drop)
        self.dropout_t_layer2 = nn.Dropout(p=layer_drop)
        self.conv_q_layer2 = nn.Conv1d(layer2_up, 4, kernel_size=1, bias=False)
        self.conv_t_layer2 = nn.Conv1d(layer2_up, 3, kernel_size=1, bias=False)

        self.pos_embed_layer2 = nn.Parameter(torch.zeros(1, 1, (img_size[0]//8)*(img_size[1]//8)))
        nn.init.trunc_normal_(self.pos_embed_layer2, std=0.02)
        
        # MLP_layer for layer1
        self.img_size_layer1 = tuple(s // 4 for s in img_size)
        self.l2_weight_up_layer1 = nn.Conv2d(mlp_layer2[-1]+dim_layer1, 256, kernel_size=1, bias=False) 
        self.l2_feat_up_layer1 = nn.Conv2d(mlp1_local[-1]+dim_layer1, 256, kernel_size=1, bias=False)   

        in_ch_l1_feat = dim_layer1 + mlp1_local[-1] + 256
        layers1_feat = []
        for idx, ch in enumerate(mlp_layer1):            
            layers1_feat += [
                nn.Conv2d(in_ch_l1_feat if idx == 0 else mlp_layer1[0], ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ]
        self.mlp_layer1_feat = nn.Sequential(*layers1_feat)

        in_ch_l1_weight = dim_layer1 + mlp_layer1[-1] + 256
        layers1_weight = []
        for idx, ch in enumerate(mlp_layer1):            
            layers1_weight += [
                nn.Conv2d(in_ch_l1_weight if idx == 0 else mlp_layer1[0], ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ]
        self.mlp_layer1_weight = nn.Sequential(*layers1_weight)

        self.l1_up_conv = nn.Conv1d(mlp_layer1[-1], layer1_up, kernel_size=1, bias=False) 
        self.dropout_q_layer1 = nn.Dropout(p=layer_drop)
        self.dropout_t_layer1 = nn.Dropout(p=layer_drop)
        self.conv_q_layer1 = nn.Conv1d(layer1_up, 4, kernel_size=1, bias=False)
        self.conv_t_layer1 = nn.Conv1d(layer1_up, 3, kernel_size=1, bias=False)

        self.pos_embed_layer1 = nn.Parameter(torch.zeros(1, 1, (img_size[0]//4)*(img_size[1]//4)))
        nn.init.trunc_normal_(self.pos_embed_layer1, std=0.02)

        # MLP_layer for layer0
        self.img_size_layer0 = tuple(s // 2 for s in img_size)
        self.l1_weight_up_layer0 = nn.Conv2d(mlp_layer1[-1]+dim_layer0, 256, kernel_size=1, bias=False) 
        self.l1_feat_up_layer0 = nn.Conv2d(mlp1_local[-1]+dim_layer0, 256, kernel_size=1, bias=False)   

        in_ch_l0_feat = dim_layer0 + mlp1_local[-1] + 256
        layers0_feat = []
        for idx, ch in enumerate(mlp_layer0):            
            layers0_feat += [
                nn.Conv2d(in_ch_l0_feat if idx == 0 else mlp_layer0[0], ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ]
        self.mlp_layer0_feat = nn.Sequential(*layers0_feat)

        in_ch_l0_weight = dim_layer0 + mlp_layer0[-1] + 256
        layers0_weight = []
        for idx, ch in enumerate(mlp_layer0):            
            layers0_weight += [
                nn.Conv2d(in_ch_l0_weight if idx == 0 else mlp_layer0[0], ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ]
        self.mlp_layer0_weight = nn.Sequential(*layers0_weight)

        self.l0_up_conv = nn.Conv1d(mlp_layer0[-1], layer0_up, kernel_size=1, bias=False)
        self.dropout_q_layer0 = nn.Dropout(p=layer_drop)
        self.dropout_t_layer0 = nn.Dropout(p=layer_drop)
        self.conv_q_layer0 = nn.Conv1d(layer0_up, 4, kernel_size=1, bias=False)
        self.conv_t_layer0 = nn.Conv1d(layer0_up, 3, kernel_size=1, bias=False)

        self.pos_embed_layer0 = nn.Parameter(torch.zeros(1, 1, (img_size[0]//2)*(img_size[1]//2)))
        nn.init.trunc_normal_(self.pos_embed_layer0, std=0.02)
                   
    def pyramid_downsample_flow(self, optical_flow, n_levels=4):
        B, C, H, W = optical_flow.shape
        flows = []
        for lvl in range(n_levels):
            scale = 2 ** (lvl + 1)
            H_i = H // scale
            W_i = W // scale
            f_i = F.interpolate(
                optical_flow,
                size=(H_i, W_i),
                mode='bilinear',
                align_corners=True
            )
            f_i = f_i / scale
            flows.append(f_i)
        return flows
    
    def flow_consistency_map(self, optical_flow12, optical_flow21, large_val=1e4, eps=1e-6):
        device = optical_flow12.device
        B, _, H, W = optical_flow12.shape

        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=optical_flow12.dtype),
            torch.arange(W, device=device, dtype=optical_flow12.dtype),
            indexing='ij'
        )
        grid = torch.stack((x, y), dim=0) # [2, H, W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1) # [B,2,H,W]

        coords_t1 = grid + optical_flow12 
        x_t1, y_t1 = coords_t1[:,0], coords_t1[:,1]

        in_bounds = (x_t1 >= 0) & (x_t1 <= W-1) & (y_t1 >= 0) & (y_t1 <= H-1)
        in_bounds = in_bounds.unsqueeze(1)         # [B,1,H,W]

        x_norm = 2.0 * x_t1 / (W - 1) - 1.0
        y_norm = 2.0 * y_t1 / (H - 1) - 1.0
        grid_norm = torch.stack((x_norm, y_norm), dim=-1)   # [B,H,W,2]
        optical_flow21_warp = F.grid_sample(
            optical_flow21, grid_norm,
            mode='bilinear', padding_mode='zeros',
            align_corners=True
        )             
        diff = optical_flow12 + optical_flow21_warp                     # [B,2,H,W] 
        D = torch.norm(diff, dim=1, keepdim=True)          # [B,1,H,W]

        M = D.clone()
        M[~in_bounds] = large_val

        max_valid = M[in_bounds].max()
        M[~in_bounds] = max_valid

        min_d, max_d = M.min(), M.max()
        M_norm = (M - min_d) / (max_d - min_d + eps)
        M_conf = 1.0 - M_norm # b, 1, h, w

        return M_conf


    def forward(self, img_t1, img_t2):
        optical_flow_ori12 = compute_optical_flow(self.raft, img_t1, img_t2, device=img_t1.device)
        optical_flow_ori21 = compute_optical_flow(self.raft, img_t2, img_t1, device=img_t1.device)

        img1_feat_0, img1_feat_1, img1_feat_2, img1_feat_3 = self.img_backbone(img_t1)
        img2_feat_0, img2_feat_1, img2_feat_2, img2_feat_3 = self.img_backbone(img_t2)

        optical_flow12_0, optical_flow12_1, optical_flow12_2, optical_flow12_3 = self.pyramid_downsample_flow(optical_flow_ori12, n_levels=4)
        optical_flow21_0, optical_flow21_1, optical_flow21_2, optical_flow21_3 = self.pyramid_downsample_flow(optical_flow_ori21, n_levels=4)

        ######################### Layer 3 ########################
        layer3_fused_feat = self.FGA_layer3(img1_feat_3, img2_feat_3, optical_flow12_3)

        layer3_concat_feat = torch.cat([img1_feat_3, layer3_fused_feat], dim=1) # b, c1+c2, h, w
        l3_concat_feat_conv = self.mlp_layer3(layer3_concat_feat) # b, mlp_layer3[-1], h, w

        B, C_3, H_3, W_3 = l3_concat_feat_conv.shape
        l3_flow_mask = self.flow_consistency_map(optical_flow12_3, optical_flow21_3) # b, 1, h, w
        masked_logits_layer3 = l3_concat_feat_conv * l3_flow_mask  # b, mlp_layer3[-1], h, w

        #masked_logits_layer3 = l3_concat_feat_conv

        masked_flat_layer3 = masked_logits_layer3.view(B, C_3, -1) # b, c, h*w
        masked_flat_layer3 = masked_flat_layer3 + self.pos_embed_layer3
        l3_weight   = torch.softmax(masked_flat_layer3, dim=2).view(B, C_3, H_3, W_3)

        l3_feat   = (layer3_fused_feat*l3_weight).sum(dim=(2,3))
        l3_up_feat = self.l3_up_conv(l3_feat.unsqueeze(-1)).squeeze(-1) # b, c

        l3_q_feat = self.dropout_q_layer3(l3_up_feat)
        l3_t_feat = self.dropout_t_layer3(l3_up_feat)

        l3_q_coarse = self.conv_q_layer3(l3_q_feat.unsqueeze(-1)).squeeze(-1)
        l3_q_coarse_norm = F.normalize(l3_q_coarse, p=2, dim=1, eps=1e-10)

        l3_t_coarse = self.conv_t_layer3(l3_t_feat.unsqueeze(-1)).squeeze(-1)


        ######################### layer 2 ###########################
        layer2_fused_feat = self.FGA_layer2(img1_feat_2, img2_feat_2, optical_flow12_2)

        layer3_weight_sr = F.interpolate(
                            l3_concat_feat_conv,
                            size=self.img_size_layer2,
                            mode='bilinear',
                            align_corners=True) # b, c, h, w

        layer3_weight_cr = self.l3_weight_up_layer2(torch.cat([layer3_weight_sr, img1_feat_2], dim=1))
        
        layer3_feat_sr = F.interpolate(
                            layer3_fused_feat,
                            size=self.img_size_layer2,
                            mode='bilinear',
                            align_corners=True)
        
        layer3_feat_cr = self.l3_feat_up_layer2(torch.cat([layer3_feat_sr, img1_feat_2], dim=1))
        
        layer2_feat = torch.cat([img1_feat_2, layer3_feat_cr, layer2_fused_feat], dim=1)
        layer2_feat_conv = self.mlp_layer2_feat(layer2_feat)

        layer2_weight = torch.cat([img1_feat_2, layer3_weight_cr, layer2_feat_conv], dim=1)
        layer2_weight_conv = self.mlp_layer2_weight(layer2_weight)

        B, C_2, H_2, W_2 = layer2_weight_conv.shape
        l2_flow_mask = self.flow_consistency_map(optical_flow12_2, optical_flow21_2) # b, 1, h, w
        masked_logits_layer2 = layer2_weight_conv * l2_flow_mask  # b, mlp_layer3[-1], h, w

        #masked_logits_layer2 = layer2_weight_conv   # b, mlp_layer3[-1], h, w

        masked_flat_layer2 = masked_logits_layer2.view(B, C_2, -1) # b, c, h*w
        masked_flat_layer2 = masked_flat_layer2 + self.pos_embed_layer2 # b, c, h*w
        l2_weight = torch.softmax(masked_flat_layer2, dim=2).view(B, C_2, H_2, W_2)
        
        l2_feat   = (layer2_feat_conv*l2_weight).sum(dim=(2,3)) # b, c
        l2_up_feat = self.l2_up_conv(l2_feat.unsqueeze(-1)).squeeze(-1)

        l2_q_feat = self.dropout_q_layer2(l2_up_feat)
        l2_t_feat = self.dropout_t_layer2(l2_up_feat)

        l2_q_coarse = self.conv_q_layer2(l2_q_feat.unsqueeze(-1)).squeeze(-1)
        l2_q_coarse_norm = F.normalize(l2_q_coarse, p=2, dim=1, eps=1e-10)

        l2_t_coarse = self.conv_t_layer2(l2_t_feat.unsqueeze(-1)).squeeze(-1)

        l2_t_quat_res = torch.cat([torch.zeros_like(l3_t_coarse[:, :1]), l3_t_coarse], dim=1) # b,4
        # q*(0,t)*q-1
        l2_q_inv_res = quat_inv(l3_q_coarse_norm) 

        l2_t_tmp_res = quat_mul(l3_q_coarse_norm, l2_t_quat_res)
        l2_t_trans_res = quat_mul(l2_t_tmp_res, l2_q_inv_res) # b,4
        l2_t_trans_res = l2_t_trans_res[:, 1:] # b,3

        l2_q_fine = quat_mul(l2_q_coarse_norm, l3_q_coarse_norm)
        
        l2_t_fine = l2_t_coarse + l2_t_trans_res


        ########################## layer 1 ############################
        layer1_fused_feat = self.FGA_layer1(img1_feat_1, img2_feat_1, optical_flow12_1)

        layer2_weight_sr = F.interpolate(
                            layer2_weight_conv,
                            size=self.img_size_layer1,
                            mode='bilinear',
                            align_corners=True) # b, c, h, w

        layer2_weight_cr = self.l2_weight_up_layer1(torch.cat([layer2_weight_sr, img1_feat_1], dim=1))
        
        layer2_feat_sr = F.interpolate(
                            layer2_feat_conv,
                            size=self.img_size_layer1,
                            mode='bilinear',
                            align_corners=True)
        
        layer2_feat_cr = self.l2_feat_up_layer1(torch.cat([layer2_feat_sr, img1_feat_1], dim=1))

        layer1_feat = torch.cat([img1_feat_1, layer2_feat_cr, layer1_fused_feat], dim=1)
        layer1_feat_conv = self.mlp_layer1_feat(layer1_feat)

        layer1_weight = torch.cat([img1_feat_1, layer2_weight_cr, layer1_feat_conv], dim=1)
        layer1_weight_conv = self.mlp_layer1_weight(layer1_weight)

        B, C_1, H_1, W_1 = layer1_weight_conv.shape
        l1_flow_mask = self.flow_consistency_map(optical_flow12_1, optical_flow21_1) # b, 1, h, w
        masked_logits_layer1 = layer1_weight_conv * l1_flow_mask  # b, mlp_layer3[-1], h, w

        #masked_logits_layer1 = layer1_weight_conv

        masked_flat_layer1 = masked_logits_layer1.view(B, C_1, -1) # b, c, h*w
        masked_flat_layer1 = masked_flat_layer1 + self.pos_embed_layer1 # b, c, h*w
        l1_weight = torch.softmax(masked_flat_layer1, dim=2).view(B, C_1, H_1, W_1)

        l1_feat   = (layer1_feat_conv*l1_weight).sum(dim=(2,3)) # b, c
        l1_up_feat = self.l1_up_conv(l1_feat.unsqueeze(-1)).squeeze(-1)

        l1_q_feat = self.dropout_q_layer1(l1_up_feat)
        l1_t_feat = self.dropout_t_layer1(l1_up_feat)

        l1_q_coarse = self.conv_q_layer1(l1_q_feat.unsqueeze(-1)).squeeze(-1)
        l1_q_coarse_norm = F.normalize(l1_q_coarse, p=2, dim=1, eps=1e-10)

        l1_t_coarse = self.conv_t_layer1(l1_t_feat.unsqueeze(-1)).squeeze(-1)

        l1_t_quat_res = torch.cat([torch.zeros_like(l2_t_fine[:, :1]), l2_t_fine], dim=1) # b,4
        # q*(0,t)*q-1
        l1_q_inv_res = quat_inv(l2_q_fine) 

        l1_t_tmp_res = quat_mul(l2_q_fine, l1_t_quat_res)
        l1_t_trans_res = quat_mul(l1_t_tmp_res, l1_q_inv_res) # b,4
        l1_t_trans_res = l1_t_trans_res[:, 1:] # b,3

        l1_q_fine = quat_mul(l1_q_coarse_norm, l2_q_fine)
        
        l1_t_fine = l1_t_coarse + l1_t_trans_res

        ########################## layer 0 ############################
        layer0_fused_feat = self.FGA_layer0(img1_feat_0, img2_feat_0, optical_flow12_0)

        layer1_weight_sr = F.interpolate(
                            layer1_weight_conv,
                            size=self.img_size_layer0,
                            mode='bilinear',
                            align_corners=True) # b, c, h, w

        layer1_weight_cr = self.l1_weight_up_layer0(torch.cat([layer1_weight_sr, img1_feat_0], dim=1))
        
        layer1_feat_sr = F.interpolate(
                            layer1_feat_conv,
                            size=self.img_size_layer0,
                            mode='bilinear',
                            align_corners=True)
        
        layer1_feat_cr = self.l1_feat_up_layer0(torch.cat([layer1_feat_sr, img1_feat_0], dim=1))

        layer0_feat = torch.cat([img1_feat_0, layer1_feat_cr, layer0_fused_feat], dim=1)
        layer0_feat_conv = self.mlp_layer0_feat(layer0_feat)

        layer0_weight = torch.cat([img1_feat_0, layer1_weight_cr, layer0_feat_conv], dim=1)
        layer0_weight_conv = self.mlp_layer0_weight(layer0_weight)

        B, C_0, H_0, W_0 = layer0_weight_conv.shape
        l0_flow_mask = self.flow_consistency_map(optical_flow12_0, optical_flow21_0) # b, 1, h, w
        masked_logits_layer0 = layer0_weight_conv * l0_flow_mask  # b, mlp_layer3[-1], h, w

        #masked_logits_layer0 = layer0_weight_conv

        masked_flat_layer0 = masked_logits_layer0.view(B, C_0, -1) # b, c, h*w
        masked_flat_layer0 = masked_flat_layer0 + self.pos_embed_layer0 # b, c, h*w
        l0_weight = torch.softmax(masked_flat_layer0, dim=2).view(B, C_0, H_0, W_0)

        l0_feat   = (layer0_feat_conv*l0_weight).sum(dim=(2,3)) # b, c
        l0_up_feat = self.l0_up_conv(l0_feat.unsqueeze(-1)).squeeze(-1)

        l0_q_feat = self.dropout_q_layer0(l0_up_feat)
        l0_t_feat = self.dropout_t_layer0(l0_up_feat)

        l0_q_coarse = self.conv_q_layer0(l0_q_feat.unsqueeze(-1)).squeeze(-1)
        l0_q_coarse_norm = F.normalize(l0_q_coarse, p=2, dim=1, eps=1e-10)

        l0_t_coarse = self.conv_t_layer0(l0_t_feat.unsqueeze(-1)).squeeze(-1)

        l0_t_quat_res = torch.cat([torch.zeros_like(l1_t_fine[:, :1]), l1_t_fine], dim=1) # b,4
        # q*(0,t)*q-1
        l0_q_inv_res = quat_inv(l1_q_fine) 

        l0_t_tmp_res = quat_mul(l1_q_fine, l0_t_quat_res)
        l0_t_trans_res = quat_mul(l0_t_tmp_res, l0_q_inv_res) # b,4
        l0_t_trans_res = l0_t_trans_res[:, 1:] # b,3

        l0_q_fine = quat_mul(l0_q_coarse_norm, l1_q_fine)
        
        l0_t_fine = l0_t_coarse + l0_t_trans_res

        q_layer0_norm = F.normalize(l0_q_fine, p=2, dim=1, eps=1e-10)
        q_layer1_norm = F.normalize(l1_q_fine, p=2, dim=1, eps=1e-10)
        q_layer2_norm = F.normalize(l2_q_fine, p=2, dim=1, eps=1e-10)
        q_layer3_norm = l3_q_coarse_norm

        return q_layer0_norm, l0_t_fine, q_layer1_norm, l1_t_fine, q_layer2_norm, l2_t_fine, q_layer3_norm, l3_t_coarse


