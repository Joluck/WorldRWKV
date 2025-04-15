import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.RWKV_7 import RWKV7Block


class Model_RWKV7(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        padding = 0

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, padding, self.dropout)

        # Decoder-only Transformer: Refer to issue: https://github.com/thuml/Large-Time-Series-Model/issues/23
        self.decoder = RWKV7Block(
                dim=configs.d_model,
                block_id=0,
                n_blocks=configs.e_layers)
        # Prediction Head
        self.proj = nn.Linear(self.d_model, configs.patch_len, bias=True)