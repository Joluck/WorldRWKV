import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModel


class Siglip2Encoder(nn.Module):
    
    def __init__(
        self,
        encoder_path,
        project_dim,
        train_mode="adapter",
        device="cuda",) -> None:
        super(Siglip2Encoder, self).__init__()

        
        self.device = device
        self.model = AutoModel.from_pretrained(encoder_path).vision_model
        self.encoder_dim = 768  #self.model.config.hidden_size

        # self.adapter = VisualAdapter(self.encoder_dim, project_dim)
    def forward(self, x):
        x = self.model(x, output_hidden_states=True).last_hidden_state
        return x


if __name__ == "__main__":
    from PIL import Image
    encoder_config = {
        'encoder_path': '/home/rwkv/models/siglip2',
        'project_dim' : 768
    }
    siglip = Siglip2Encoder(**encoder_config)
    image1 = Image.open('/home/rwkv/data/vision_step2/data/chartqa/train/png/34.png').convert('RGB')
    image2 = Image.open('/home/rwkv/data/vision_step2/data/chartqa/train/png/43.png').convert('RGB')
    images = [image1, image2]
    y = siglip(images)
    print(y.shape)