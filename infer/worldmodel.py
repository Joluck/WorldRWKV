
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


from world.world_encoder import WorldEncoder
from framefusion.siglip_adapter import apply_siglip_framefusion

from typing import Union, Optional, Dict, Any
from collections.abc import Sequence
from PIL import Image

class Worldinfer():
    def __init__(self, model_path, encoder_type, encoder_path, strategy='cuda bf16', args=None, 
                 use_token_reduction=False, token_reduction_params=None):
        """
        Initialize a Worldinfer model with optional token reduction.
        
        Args:
            model_path: Path to the RWKV model
            encoder_type: Type of encoder to use (e.g., 'siglip')
            encoder_path: Path to the encoder model
            strategy: RWKV strategy string (e.g., 'cuda bf16')
            args: PIPELINE_ARGS for RWKV
            use_token_reduction: Whether to use token reduction (FrameFusion)
            token_reduction_params: Parameters for token reduction, a dict with keys:
                - cost: The computational budget (higher values allow more token reduction)
                - similarity_threshold: Threshold for token similarity to be merged
                - ratio_threshold: Minimum ratio of tokens to keep
                - for_single_images: Whether to apply reduction to single images
        """
        # Set up token reduction parameters
        self.use_token_reduction = use_token_reduction
        self.token_reduction_params = {
            'cost': 0.3,
            'similarity_threshold': 0.6,
            'ratio_threshold': 0.1,
            'for_single_images': True
        }
        if token_reduction_params is not None:
            self.token_reduction_params.update(token_reduction_params)
        
        # Store encoder type for later use
        self.encoder_type = encoder_type

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

        if args==None:
            self.args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.0, top_k=0, # top_k = 0 then ignore
                                alpha_frequency = 0.0,
                                alpha_presence = 0.0,
                                token_ban = [0], # ban the generation of some tokens
                                token_stop = [24], # stop generation whenever you see any token here
                                chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
        else:
            self.args=args
        print('RWKV finish!!!')

        config = {
            'encoder_type': encoder_type,
            'encoder_path': encoder_path,
            'project_dim' : n_embd
        }
        self.modality = WorldEncoder(**config).to('cuda', torch.bfloat16)        
        self.modality.load_checkpoint(modality_dict)
        
        # Apply token reduction if requested
        if self.use_token_reduction:
            self.apply_token_reduction()
    
    def apply_token_reduction(self):
        """
        Apply token reduction to the encoder based on the configured parameters.
        Currently only supports SIGLIP encoder.
        """
        if self.encoder_type.lower() == 'siglip':
            # Get the SIGLIP encoder from the model
            siglip_encoder = self.modality.world_encoder
            
            # Apply FrameFusion to the encoder
            apply_siglip_framefusion(
                siglip_encoder, 
                cost=self.token_reduction_params['cost'],
                similarity_lower_bound=self.token_reduction_params['similarity_threshold'],
                ratio_lower_bound=self.token_reduction_params['ratio_threshold'],
                for_single_images=self.token_reduction_params['for_single_images']
            )
            print(f"Applied token reduction with parameters: {self.token_reduction_params}")
        else:
            print(f"Token reduction not supported for encoder type: {self.encoder_type}")


    def generate(self, text, modality:Union[str, None, Sequence[Image.Image]]='none', state=None):
        if isinstance(modality, str):
            y=None
        else:
            y = self.modality(modality).to(self.DTYPE)
        result, state = self.pipeline.generate(text, token_count=500, args=self.args, callback=None, state=state, sign=y)
        return result, state

