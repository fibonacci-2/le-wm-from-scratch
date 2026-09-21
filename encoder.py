import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import timm

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        model = self.model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def forward(self, img):
        model = self.model
        output = model.forward_features(self.transforms(img))
        output = model.forward_head(output, pre_logits=True) # 1, 192
        return output
        
class MLP(nn.Module):
    def __init__(self, input_size=192, output_size=192):
        super().__init__()
        self.proj = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(num_features=output_size)


    def forward(self, emb):
        emb = self.proj(emb)
        return self.bn(emb)


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.vit = ViT()
        self.mlp = MLP(input_size, output_size)

    def forward(self, o):
        # o: B, C, H, W
        emb = self.vit(o)
        return self.mlp(emb)
    
