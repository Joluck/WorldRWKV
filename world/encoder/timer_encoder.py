import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM

class TimerAdapter(nn.Module):
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

class TimerEncoder(nn.Module):
    def __init__(
        self,
        prediction_length,
        device="cuda",
    ):
        super(TimerEncoder, self).__init__()
        self.device = device
        self.prediction_length = prediction_length

        self.model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True).to(self.device)

            

    def forward(self, x):
        x = self.model.generate(x, max_new_tokens=self.prediction_length)
        
        return x


if __name__ == "__main__":
    encoder = TimerEncoder(prediction_length=96)
    batch_size, lookback_length = 1, 2880
    x = torch.randn(batch_size, lookback_length).cuda()
    x = encoder(x)
    print(x.shape)
    print("timer encoder test pass")
