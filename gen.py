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

from world.speech_encoder import SpeechEncoder
from world.visual_encoder import VisualEncoder

class Worldinfer():
    def __init__(self, model_path, modality_path='/home/rwkv/JL/audio', strategy='cuda bf16'):

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
                k = key.replace('modality.', '')
                modality_dict[k] = value 
        model = RWKV(model=self.model_weight, strategy=strategy)
        self.pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


        self.args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.0, top_k=0, # top_k = 0 then ignore
                            alpha_frequency = 0.0,
                            alpha_presence = 0.0,
                            token_ban = [0], # ban the generation of some tokens
                            token_stop = [24], # stop generation whenever you see any token here
                            chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
        print('RWKV finish!!!')
        # self.modality = SpeechEncoder(
        #     modality_path,
        #     n_embd,
        #     downsample_K=5,
        #     hidden_dim=2048,
        #     train_mode="adapter",
        #     device='cuda',
        # ).to('cuda', torch.bfloat16)

        self.modality = VisualEncoder(
            modality_path,
            n_embd,
        )
        self.modality.eval()
        
        self.modality.load_state_dict(modality_dict, strict=False)

        self.msg = '\x16Assistant:'


    def generate(self, audio):
        y= self.modality(audio).to(self.DTYPE)
        result = self.pipeline.generate(self.msg, token_count=500, args=self.args, callback=None, state=None, sign=y)
        return result
    
    def nlp_gen(self, ctx):
        result = self.pipeline.generate(ctx, token_count=500, args=self.args, callback=None, state=None)
        return result
    
    def prefill(self, ctx, audio):
        y= self.modality(audio).to(self.DTYPE)
        result = self.pipeline.prefill(ctx, token_count=500, args=self.args, callback=None, state=None, sign=y)
        return result


# import librosa
# from datasets import load_dataset

# worldinfer = Worldinfer(model_path='/home/rwkv/JL/out_model/asr-step2ttt/rwkv-0')

# dataset = load_dataset('/home/rwkv/JL/data/fixie-ai-librispeech_asr/clean')
# data = dataset['test']
# print(len(data))

# for idx in range(100):
#     sample = data[idx]
#     audio = sample['audio']
#     data_answer = sample['text'] #####caption
#     audio = librosa.resample(audio['array'],orig_sr= audio['sampling_rate'],target_sr= 16000)  # sr=None 保持原采样率
#     res = worldinfer.generate(audio)
#     print('ori:  ', data_answer)
#     print('res: ', res)