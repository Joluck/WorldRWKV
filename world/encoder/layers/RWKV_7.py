from rwkvfla.layers import RWKV7Attention # type: ignore
from rwkvfla.utils import device
from torch import nn
import torch

torch.set_float32_matmul_precision('high')

class TMix(nn.Module):
    def __init__(self, dim, block_id, n_blocks):
        super().__init__()
        self.rwkv7 = RWKV7Attention(
            "chunk",
            dim,
            layer_idx=block_id
        )
    def forward(self, x, v_first):
        x_attn, _, past_key_values, v_first = self.rwkv7(x, v_first=v_first)
        return x_attn, v_first
class CMix(nn.Module):
    def __init__(self, dim, hidden_dim, block_id, n_blocks):
        super().__init__()
        self.value = nn.Linear(dim, dim)
    def forward(self, x):
        return self.value(x)
class RWKV7Block(nn.Module):
    def __init__(self, dim, block_id, n_blocks):
        super().__init__()
        self.attn = TMix(dim, block_id, n_blocks)
        self.mlp = CMix(dim, dim * 4, block_id, n_blocks)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        x_attn, v_first = self.attn(self.norm1(x), v_first=None)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x, v_first
    
if __name__ == "__main__":
    dim = 512  # example dimension
    block_id = 0
    n_blocks = 12
    v_first = None

    rwkv_block = RWKV7Block(dim, block_id, n_blocks).to("cuda")
    
    # Dummy input tensor
    x = torch.randn(1, 10, dim).to("cuda")

    # Forward pass
    x, _ = rwkv_block(x)
    print(x.shape)
    print(x)