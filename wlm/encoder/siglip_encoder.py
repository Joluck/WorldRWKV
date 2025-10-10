import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModel, SiglipImageProcessor, AutoConfig

class SiglipEncoder(nn.Module):
    
    def __init__(
        self,
        config,
        train_mode="adapter",
        device="cuda",) -> None:
        super(SiglipEncoder, self).__init__()
        self.encoder_path = config.encoder_path
        self.device = device
        config_obj = AutoConfig.from_pretrained(self.encoder_path)
        self.vision_model = AutoModel.from_config(config_obj).vision_model
        self.image_processor = SiglipImageProcessor.from_pretrained(self.encoder_path)
        self.encoder_dim = 768  #self.model.config.hidden_size
        # self.adapter = VisualAdapter(self.encoder_dim, project_dim)

    def forward(self, x):
        x= self.image_processor(x, return_tensors="pt")['pixel_values'].to(self.device,dtype=torch.bfloat16)
        x = self.vision_model(x, output_hidden_states=True).last_hidden_state
        return x

if __name__ == "__main__":
    from PIL import Image
    from types import SimpleNamespace
    encoder_config = {
        'encoder_path': '/home/rwkv/models/siglip',
        'project_dim' : 768
    }
    encoder_config = SimpleNamespace(**encoder_config)
    siglip = SiglipEncoder(encoder_config)
    print(siglip)
    image1 = Image.open('/home/rwkv/data/vision_step2/data/chartqa/train/png/34.png').convert('RGB')
    image2 = Image.open('/home/rwkv/data/vision_step2/data/chartqa/train/png/43.png').convert('RGB')
    images = [image1, image2]
    y = siglip(images)
    print(y.shape)