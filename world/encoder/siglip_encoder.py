import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModel, SiglipImageProcessor

class VisualAdapter(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, encoder_dim, project_dim, hidden_dim=None, use_conv=False):

        super().__init__()
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim
        self.use_conv = use_conv
        if self.hidden_dim==None:
            self.hidden_dim = project_dim*4

        self.pre_norm = nn.LayerNorm(self.project_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )
        if use_conv:
            self.conv = nn.Conv1d(
                    in_channels=encoder_dim,
                    out_channels=encoder_dim,
                    bias=False,
                    kernel_size=3,
                    stride=2
            )

    
    def forward(self, x):
        if self.use_conv:
            x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        x = self.mlp(x)
        return x + self.pre_norm(x)





class SiglipEncoder(nn.Module):
    
    def __init__(
        self,
        encoder_path,
        project_dim,
        train_mode="adapter",
        device="cuda",) -> None:
        super(SiglipEncoder, self).__init__()

        
        self.device = device
        self.model = AutoModel.from_pretrained(encoder_path).vision_model
        self.image_processor = SiglipImageProcessor.from_pretrained(encoder_path)
        self.encoder_dim = 768  #self.model.config.hidden_size

        # self.adapter = VisualAdapter(self.encoder_dim, project_dim)
    def forward(self, x):
        x= self.image_processor(x, return_tensors="pt")['pixel_values'].to(self.device,dtype=torch.bfloat16)
        x = self.model(x, output_hidden_states=True).last_hidden_state
        return x


if __name__ == "__main__":
    from PIL import Image
    encoder_config = {
        'encoder_path': '/home/rwkv/models/siglip2',
        'project_dim' : 768
    }
    siglip = SiglipEncoder(**encoder_config)
    image1 = Image.open('/home/rwkv/data/vision_step2/data/chartqa/train/png/34.png').convert('RGB')
    image2 = Image.open('/home/rwkv/data/vision_step2/data/chartqa/train/png/43.png').convert('RGB')
    images = [image1, image2]
    y = siglip(images)
    print(y.shape)