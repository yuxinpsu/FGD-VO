import torch 
import torch.nn as nn
import timm

class PyramidBackbone(nn.Module):
    def __init__(self, HRNet_Name='hrnet_w32', HRNet_Pretrained=True):
        super(PyramidBackbone, self).__init__()

        self.hrnet = self.Load_HRNet(HRNet_Name, HRNet_Pretrained)


    def Load_HRNet(self, Name, Pretrained):
        model = timm.create_model(
            model_name=Name,
            pretrained=Pretrained,
            features_only=True,  
            out_indices=(0, 1, 2, 3),  
        )
        return model
    
    def forward(self, x):
        features = self.hrnet(x)
        return features
    
if __name__ == "__main__":
    backbone = PyramidBackbone()
    
    x = torch.randn(1, 3, 2400, 2400)
    features = backbone(x)
    
    for i, feat in enumerate(features):
        print(f"Feature {i}: {feat.shape}")

    print(features[-4].shape)