import torch
import torch.nn as nn
import torch.nn.functional as F

def quat_mul(q, r):
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)


def quat_inv(q, eps=1e-10):
    q_conj = q.clone()
    q_conj[:, 1:] = -q_conj[:, 1:]
    norm_sq = q.pow(2).sum(-1, keepdim=True).clamp_min(eps)
    return q_conj / norm_sq


class FGVONetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_x = nn.Parameter(torch.tensor(0.0))
        self.w_q = nn.Parameter(torch.tensor(-2.5))
        self.layer_weights = [0.2, 0.4, 0.8, 1.6]

    def normalize_quat(self, q, eps=1e-10):
        norm = torch.norm(q, dim=-1, keepdim=True).clamp_min(eps)
        return q / norm

    def forward(self,
                l0_q, l0_t,
                l1_q, l1_t,
                l2_q, l2_t,
                l3_q, l3_t,
                q_gt, t_gt):
        losses = []
        raw_q_losses = []
        raw_t_losses = []

        for (l_q, l_t), weight in zip(
            [(l0_q, l0_t), (l1_q, l1_t),
             (l2_q, l2_t), (l3_q, l3_t)],
            self.layer_weights
        ):
            l_q_norm = self.normalize_quat(l_q)
 
            loss_q = ((q_gt - l_q_norm).pow(2).sum(dim=-1)+ 1e-10).mean()                          

            loss_x = (l_t - t_gt).norm(p=1, dim=-1).mean()  

            raw_q_losses.append(weight*loss_q)
            raw_t_losses.append(weight*loss_x)

            level_loss = (
                loss_x * torch.exp(-self.w_x) + self.w_x +
                loss_q * torch.exp(-self.w_q) + self.w_q
            )
            # level_loss = loss_x + loss_q*10
            losses.append(weight * level_loss)

        total_loss = sum(losses)
        total_raw_q_loss = sum(raw_q_losses)
        total_raw_t_loss = sum(raw_t_losses)

        return total_loss, total_raw_q_loss, total_raw_t_loss             



class FlowGuidedAttention(nn.Module):

    def __init__(self, nsample_coarse=32, nsample_fine=9, C_in=128, mlp1=(128, 64, 64), mlp2=(128, 64),
                 mlp1_local=(64, 64), mlp2_local=(64,64), geo_hiden=64):
        super().__init__()

        self.nsample_coarse = nsample_coarse
        self.nsample_fine = nsample_fine
        # MLP_layer1_global
        layers1 = []
        in_ch = 3 + 2*C_in
        for ch in mlp1:                    
            layers1 += [nn.Conv2d(in_ch, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
            in_ch = ch
        self.mlp1 = nn.Sequential(*layers1)

        # Encoder_layer_global
        self.geo_enc1 = nn.Conv2d(3, geo_hiden, 1)

        # MLP_layer2_global
        layers2 = []
        in_ch = mlp1[-1] + geo_hiden
        for ch in mlp2:                       
            layers2 += [nn.Conv2d(in_ch, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
            in_ch = ch
        self.mlp2 = nn.Sequential(*layers2)

        # Weight_layer_global
        self.global_weight = nn.Conv2d(mlp2[-1], 1, 1)

        # offset predictor global
        C_in_offset_global = C_in*2
        self.offset_predictor_global = nn.Conv2d(C_in_offset_global, self.nsample_coarse*2, kernel_size=3, padding=1)

        # MLP_layer1_local
        layers1_local = []
        in_ch = 3 + 2*mlp1[-1]
        for ch in mlp1_local:                    
            layers1_local += [nn.Conv2d(in_ch, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
            in_ch = ch
        self.mlp1_local = nn.Sequential(*layers1_local)

        # Encoder_layer_local
        self.geo_enc1_local = nn.Conv2d(3, geo_hiden, 1)

        # MLP_layer2_local
        layers2_local = []
        in_ch = mlp1_local[-1] + geo_hiden
        for ch in mlp2_local:                       
            layers2_local += [nn.Conv2d(in_ch, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
            in_ch = ch
        self.mlp2_local = nn.Sequential(*layers2_local)

        # Weight_layer_local
        self.global_weight_local = nn.Conv2d(mlp2_local[-1], 1, 1)

        # offset predictor local
        self.offset_predictor_local = nn.Conv2d(mlp1[-1], self.nsample_fine*2, kernel_size=3, padding=1)



    def forward(self, img1, img2, optical_flow12):
        B, C, H, W = img1.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )

        # [B, H, W, 2]
        grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        grid = grid.type_as(img1)  

        target_coords = grid + optical_flow12.permute(0, 2, 3, 1)

        feat_global = torch.cat([img1, img2], dim=1)
        offset_global = self.offset_predictor_global(feat_global)
        offset_global = offset_global.view(B, self.nsample_coarse, 2, H, W).permute(0, 3, 4, 1, 2) #[B, H, W, K, 2]
        img2_sample_coords = target_coords.unsqueeze(3) + offset_global  # [B, H, W, K, 2]
        #img2_sample_coords = sample_features_from_coords(target_coords, num_samples=self.nsample_coarse) #[B,H,W,K,2]

        # Feats sampling from img2
        x_norm =  2.0 * img2_sample_coords[..., 0] / (W - 1) - 1.0
        y_norm =  2.0 * img2_sample_coords[..., 1] / (H - 1) - 1.0
        grid_norm    = torch.stack([x_norm, y_norm], dim=-1)  # [B,H,W,K,2]
        grid_norm = grid_norm.view(B, H, W * self.nsample_coarse, 2)

        img2_sample_feats_flat = F.grid_sample(
            img2, grid_norm,
            mode='bilinear',          
            padding_mode='border',   
            align_corners=True
        )

        img2_sample_feats = img2_sample_feats_flat.view(B, C, H, W, self.nsample_coarse).permute(0, 2, 3, 4, 1) # [B,H,W,K,d]

        img1_coords = grid.unsqueeze(3).repeat(1, 1, 1, self.nsample_coarse, 1)
        img1_feats = img1.permute(0, 2, 3, 1).unsqueeze(3).expand(-1, -1, -1, self.nsample_coarse, -1)# [B,H,W,K,d]

        # Global fusion
        coords_diff = img2_sample_coords - img1_coords  # [B,H,W,K,2]
        distance = torch.norm(coords_diff, dim=-1, keepdim=True) # [B,H,W,K,1]
        geo_information = torch.cat([coords_diff, distance], dim=-1) # [B,H,W,K,3]

        feats_information = torch.cat([img1_feats, img2_sample_feats], dim=-1) # [B,H,W,K,2d]

        feats = torch.cat([geo_information, feats_information], dim=-1) # [B,H,W,K,3+2d]

        BHW = B * H * W
        feat_conv = feats.view(BHW, self.nsample_coarse, -1).contiguous().transpose(1, 2).unsqueeze(-1)  # [B*H*W,3+2d,K,1]   
        feat_conv = self.mlp1(feat_conv)  # [B*H*W, mlp1[-1], K, 1]  

        geom_conv = geo_information.view(BHW, self.nsample_coarse, -1).contiguous().transpose(1, 2).unsqueeze(-1)  # [B*H*W,3,K,1]
        geom_conv = self.geo_enc1(geom_conv)

        feat_weight = torch.cat([geom_conv, feat_conv], dim=1)  
        feat_weight = self.mlp2(feat_weight)

        score = self.global_weight(feat_weight).squeeze(3)
        attention_score = torch.softmax(score, dim=-1).unsqueeze(-1) # [BHW, 1, K, 1]

        fused_feat = (attention_score * feat_conv).sum(dim=2).squeeze(-1).view(B, H, W, -1) # [B,H,W,mlp1[-1]]

        # Local fusion
        _, _, _, C_local = fused_feat.shape
        fused_feat_img1 = fused_feat.permute(0, 3, 1, 2) # [B,mlp1[-1],H,W]

        offset_local = self.offset_predictor_local(fused_feat_img1)
        offset_local = offset_local.view(B, self.nsample_fine, 2, H, W).permute(0, 3, 4, 1, 2) #[B, H, W, K, 2]
        target_coords_local = grid.unsqueeze(3) + offset_local #[B, H, W, K, 2]

        x_norm_local =  2.0 * target_coords_local[..., 0] / (W - 1) - 1.0
        y_norm_local =  2.0 * target_coords_local[..., 1] / (H - 1) - 1.0
        grid_norm_local    = torch.stack([x_norm_local, y_norm_local], dim=-1)  # [B,H,W,K,2]
        grid_norm_local = grid_norm_local.view(B, H, W * self.nsample_fine, 2)

        img1_sample_feats = F.grid_sample(
            fused_feat_img1, grid_norm_local,
            mode='bilinear',          
            padding_mode='border',   
            align_corners=True
        )

        img1_sample_feats = img1_sample_feats.view(B, C_local, H, W, self.nsample_fine).permute(0, 2, 3, 4, 1) # [B,H,W,K,d]
        # img1_sample_feats = F.unfold(fused_feat_img1, kernel_size=3, padding=1)
        # img1_sample_feats = img1_sample_feats.view(B, C_local, self.nsample_fine, H, W).permute(0, 3, 4, 2, 1) # [B,H,W,K,d]

        img1_local_feats = fused_feat_img1.permute(0, 2, 3, 1).unsqueeze(3).repeat(1, 1, 1, self.nsample_fine, 1) # [B,H,W,K,mlp1[-1]]

        distance_local = torch.norm(offset_local, dim=-1, keepdim=True) # [B,H,W,K,1]
        geo_information_local = torch.cat([target_coords_local, distance_local], dim=-1) # [B,H,W,K,3]

        feats_information_local = torch.cat([img1_local_feats, img1_sample_feats], dim=-1) # [B,H,W,K,2d]

        feats_local = torch.cat([geo_information_local, feats_information_local], dim=-1) # [B,H,W,K,3+2d]

        feat_local_conv = feats_local.view(BHW, self.nsample_fine, -1).contiguous().transpose(1, 2).unsqueeze(-1)  # [B*H*W,3+2d,K,1]   
        feat_local_conv = self.mlp1_local(feat_local_conv)  # [B*H*W, mlp1[-1], K, 1]  

        geom_conv_local = geo_information_local.view(BHW, self.nsample_fine, -1).contiguous().transpose(1, 2).unsqueeze(-1)  # [B*H*W,3,K,1]
        geom_conv_local = self.geo_enc1_local(geom_conv_local)

        feat_weight_local = torch.cat([geom_conv_local, feat_local_conv], dim=1)  
        feat_weight_local = self.mlp2_local(feat_weight_local)

        score_local = self.global_weight_local(feat_weight_local).squeeze(3)
        attention_score_local = torch.softmax(score_local, dim=-1).unsqueeze(-1) # [BHW, 1, K, 1]

        fused_feat_local_global = (attention_score_local * feat_local_conv).sum(dim=2).squeeze(-1).view(B, H, W, -1) # [B,H,W,mlp1[-1]]

        return fused_feat_local_global.permute(0, 3, 1, 2)
    


# class RAFTWarpAlign(nn.Module):
#     def __init__(self, in_channels, out_channels=64):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, feat, flow):

#         B, C, H, W = feat.shape
#         device = feat.device
#         dtype = feat.dtype

#         y, x = torch.meshgrid(
#             torch.arange(H, device=device, dtype=dtype),
#             torch.arange(W, device=device, dtype=dtype),
#             indexing='ij'
#         )

#         base_grid = torch.stack((x, y), dim=0).unsqueeze(0).expand(B, -1, -1, -1)
#         sample_grid = base_grid + flow

#         x_norm = 2.0 * sample_grid[:, 0] / (W - 1) - 1.0
#         y_norm = 2.0 * sample_grid[:, 1] / (H - 1) - 1.0
#         grid_norm = torch.stack((x_norm, y_norm), dim=-1)

#         warped_feat = F.grid_sample(
#             feat,
#             grid_norm,
#             mode='bilinear',
#             padding_mode='zeros',
#             align_corners=True
#         )

#         warped_feat = self.proj(warped_feat)
#         return warped_feat
    

# def add_gaussian_noise_to_flow(optical_flow12, optical_flow21, sigma=1.0, same_noise=False):

#     if sigma <= 0:
#         return optical_flow12, optical_flow21

#     noise12 = torch.randn_like(optical_flow12) * sigma

#     if same_noise:
#         noise21 = -noise12
#     else:
#         noise21 = torch.randn_like(optical_flow21) * sigma

#     noisy_flow12 = optical_flow12 + noise12
#     noisy_flow21 = optical_flow21 + noise21

#     return noisy_flow12, noisy_flow21

