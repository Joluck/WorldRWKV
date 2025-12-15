
import numpy as np

import os, sys, torch, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import torch
print(torch.__version__)
print(torch.version.cuda)

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
from infer.rwkv.model import RWKV # pip install rwkv
from infer.rwkv.utils import PIPELINE, PIPELINE_ARGS

from lg_train.registry import Projector_Registry, Encoder_Registry
from lg_train.prepare.custom_transformers import get_image_processor


class Worldinfer():
    def __init__(self, model_path, encoder_type, encoder_path, strategy='cuda bf16', args=None, processor=None, use_conv=False):

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
        proj_dict = {}
        llm_dict = {}
        encoder_dict = {}
        for key, value in self.model_weight.items():
            if 'emb.weight' in key:
                _, n_embd = value.shape
            if key.startswith('encoder.'):
                k = key.replace('encoder.', '', 1) 
                encoder_dict[k] = value 
            if key.startswith('proj.'):
                k = key.replace('proj.', '', 1) 
                proj_dict[k] = value 
            elif key.startswith('llm.'):
                k = key.replace('llm.', '', 1)
                llm_dict[k] = value 
        model = RWKV(model=llm_dict, strategy=strategy)
        self.pipeline = PIPELINE(model, "wr_vocab_v20230424")

        if args==None:
            self.args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.0, top_k=0, # top_k = 0 then ignore
                                alpha_frequency = 0.0,
                                alpha_presence = 0.0,
                                token_ban = [0], # ban the generation of some tokens
                                token_stop = [24], # stop generation whenever you see any token here
                                chunk_len = 1024) # split input into chunks to save VRAM (shorter -> slower)
        else:
            self.args=args
        print('RWKV finish!!!')

        config = {
            'encoder_path': encoder_path,
            'project_dim' : n_embd
        }
        self.encoder = Encoder_Registry[encoder_type] (**config).to('cuda', self.DTYPE).eval()  
        if processor is not None:
            self.image_processor = get_image_processor(768, 384, True)
            use_conv = True
        proj_config = {
            'encoder_dim': 768,
            'project_dim': n_embd,
            'use_conv': use_conv
        }
        self.proj = Projector_Registry[encoder_type] (**proj_config).to('cuda', self.DTYPE)    
        self.encoder.load_state_dict(encoder_dict)
        self.proj.load_state_dict(proj_dict)

        self.processor = processor

    def process_wr(self, text, image = None, img_token_len = 576):
        content = ''
        if image is not None:
            for i in range(len(image)):
                replacement ="<|image_pad|>" * img_token_len
                content+=replacement
        content = f'\x16User:{content}{text}\x17\x16Assistant:'
        return content
    def generate(self, text, modality=None, state=None, img_token_len = 576):
        if modality is not None:
            if self.processor is not None:
                pixel_values,_ = self.image_processor(modality)
                pixel_values = pixel_values.to(device='cuda',dtype=self.DTYPE)
                img_token_len = 115
            else:
                pixel_values = modality
            images_embeds = self.encoder(pixel_values)
            modality = self.proj(images_embeds)
        text = self.process_wr(text, modality, img_token_len)
        result, state = self.pipeline.generate(text, token_count=500, args=self.args, callback=None, state=state, sign=modality)
        return result, state


