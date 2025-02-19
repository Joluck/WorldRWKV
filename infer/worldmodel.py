
import numpy as np
from datasets import load_dataset
import jsonlines
from transformers import CLIPProcessor, CLIPModel

import os, sys, torch, time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import torch
print(torch.__version__)
print(torch.version.cuda)

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
from infer.rwkv.model import RWKV # pip install rwkv
from infer.rwkv.utils import PIPELINE, PIPELINE_ARGS


from world.world_encoder import WorldEncoder

class Worldinfer():
    def __init__(self, model_path, encoder_type, encoder_path, strategy='cuda bf16'):

        ss = strategy.split(' ')
        DEVICE = ss[0]
        if ss[1] == 'fp16':
            self.DTYPE = torch.half
        elif ss[1] == 'fp32':
            self.DTYPE = torch.float32
        elif ss[1] == 'bf16':
            self.DTYPE = torch.bfloat16
        else:
            assert False, "currently rwkv7 strategy must be: cuda/cpu fp16/fp32/bf16"
        
        self.model_weight = torch.load(model_path + '.pth', map_location=DEVICE)
        modality_dict = {}
        for key, value in self.model_weight.items():
            if 'emb.weight' in key:
                _, n_embd = value.shape
            if 'modality' in key:
                k = key.replace('modality.world_encoder.', '')
                modality_dict[k] = value 
        model = RWKV(model=self.model_weight, strategy=strategy)
        self.pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


        self.args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.4, top_k=0, # top_k = 0 then ignore
                            alpha_frequency = 0.0,
                            alpha_presence = 0.0,
                            token_ban = [0], # ban the generation of some tokens
                            token_stop = [24], # stop generation whenever you see any token here
                            chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
        print('RWKV finish!!!')

        config = {
            'encoder_type': encoder_type,
            'encoder_path': encoder_path,
            'project_dim' : n_embd
        }
        self.modality = WorldEncoder(**config).to('cuda', torch.bfloat16)        
        self.modality.load_checkpoint(modality_dict)


    def generate(self, msg, modality):
        y = self.modality(modality).to(self.DTYPE)
        result = self.pipeline.generate(msg, token_count=500, args=self.args, callback=None, state=None, sign=y)
        return result

