import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModel, SiglipImageProcessor

class VisualAdapter(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, encoder_dim, project_dim, hidden_dim=None):

        super().__init__()
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim

        if self.hidden_dim==None:
            self.hidden_dim = project_dim*2

        self.pre_norm = nn.LayerNorm(self.project_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.encoder_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )
        # self.proj = nn.Sequential(
        #     nn.Linear(self.encoder_dim, self.hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_dim, self.project_dim),
        # )

    
    def forward(self, x):        
        x = self.proj(x)
        return x + self.pre_norm(x)

class VideoAdapter(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.input_dim = input_dim

        def depthwise_conv():
            return nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, groups=input_dim)

        self.reduce_conv = nn.Sequential(
            depthwise_conv(), nn.Conv1d(input_dim, input_dim, kernel_size=1), nn.ReLU(),  # 1/2
            depthwise_conv(), nn.Conv1d(input_dim, input_dim, kernel_size=1), nn.ReLU(),  # 1/4
            # depthwise_conv(), nn.Conv1d(input_dim, input_dim, kernel_size=1), nn.ReLU(),  # 1/8
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.reduce_conv(x)
        x = x.transpose(1, 2)
        return x

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

        self.adapter = VisualAdapter(self.encoder_dim, project_dim)
        self.VideoAdapter = VideoAdapter()
    def forward(self, x):
        if isinstance(x, list):  # 输入图像列表，用于视频理解
            img_tensor_list = []
            for frame in x:
                frame = torch.from_numpy(self.image_processor(frame)['pixel_values'][0]).to(self.device,dtype=torch.bfloat16)
                frame = self.model(frame.unsqueeze(0), output_hidden_states=True).last_hidden_state
                img_tensor_list.append(frame)
            out_put = torch.cat(img_tensor_list, dim=1)
            out_put = self.adapter(out_put)
            out_put = self.VideoAdapter(out_put)
            # print(out_put.shape)
            return out_put

        else:  # 默认处理单个图像 
            x = torch.from_numpy(self.image_processor(x)['pixel_values'][0]).to(self.device,dtype=torch.bfloat16)
            x = self.model(x.unsqueeze(0), output_hidden_states=True).last_hidden_state
            x = self.adapter(x)
            return x
    