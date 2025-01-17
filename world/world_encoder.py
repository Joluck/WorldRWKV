import torch
import torch.nn as nn
import torch.nn.functional as F



class WorldEncoder(nn.Module):
    def __init__(self, args, modality):
        super().__init__()
        self.modality = modality

    def set_gradient(self, train_mode):
        """
        if train_mode is "adapter", only train the adapter layers, otherwise train the whole model
        """
        if train_mode == "adapter":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.adapter.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.adapter.parameters():
                param.requires_grad = True
    def forward(self, x):
        x = self.modality(x)
        return x