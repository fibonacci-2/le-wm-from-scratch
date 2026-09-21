import torch
import torch.nn as nn

class ViTSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'vit_small_patch16_dinov3_qkvb.eupe_lvd1689m',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        model = self.model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def forward(self, img):
        model = self.model
        output = model.forward_features(self.transforms(img))
        output = model.forward_head(output, pre_logits=True) # B, 192
        return output
        
class Predictor(nn.Module):
    def __intit__(self):
        super().__init__()
