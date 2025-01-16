import torch
import torch.nn as nn
import torch.nn.functional as F



class WorldEncoder(nn.Module):
    def __init__(self, modality, adapter):
        super().__init__()
        self.modality = modality

        self.adapter = adapter

    def forward(self, x):
        x = self.modality(x)
        x = self.adapter(x)
        return x