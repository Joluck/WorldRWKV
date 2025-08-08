########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from .block import Block
import deepspeed
class RWKV7(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)


    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, past_state=None):
        args = self.args
        
        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)
        v_first = torch.empty_like(inputs_embeds)

        for block in self.blocks:
            if args.grad_cp == 1:
                if args.train_type == 'state' or args.peft !='none':
                    inputs_embeds, v_first = torch_checkpoint(block, inputs_embeds, v_first ,attention_mask,past_state,use_reentrant=False)
                else:
                    inputs_embeds, v_first = deepspeed.checkpointing.checkpoint(block, inputs_embeds, v_first, attention_mask, past_state)
            else:
                inputs_embeds, v_first = block(inputs_embeds, v_first, attention_mask=attention_mask, past_state=past_state)

        inputs_embeds = self.ln_out(inputs_embeds)
        inputs_embeds = self.head(inputs_embeds)

        return inputs_embeds