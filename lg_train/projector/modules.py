import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipImageProcessor
from transformers.activations import GELUActivation


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



class VlProj(nn.Module):
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
            GELUActivation(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )
        self.conv = nn.Conv1d(
                in_channels=encoder_dim,
                out_channels=encoder_dim,
                bias=False,
                kernel_size=6,
                stride=5
        )

    
    def forward(self, x):
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        x = self.mlp(x)
        return x

class ModalityProjector(nn.Module):

    def __init__(self, encoder_dim, project_dim, hidden_dim=None, use_conv=False):
        super().__init__()
        self.input_dim = 12288
        self.output_dim = project_dim
        self.scale_factor = 4

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0 # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor
        
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)
        
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x