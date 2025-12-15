import os
import torch.nn as nn
from .ffn import RWKV7_CMIX
from .att import RWKV7_TMIX
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV7_TMIX(args, layer_id)  
        self.ffn = RWKV7_CMIX(args, layer_id)




    def forward(self, x, v_first, attention_mask = None, past_state=None):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first, attention_mask = attention_mask, past_state=past_state)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x), attention_mask = attention_mask)
        return x, v_first