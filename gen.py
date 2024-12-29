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
CHAT_LANG = 'Chinese'
from infer.rwkv.model import RWKV # pip install rwkv
from infer.rwkv.utils import PIPELINE, PIPELINE_ARGS

model = RWKV(model='/home/rwkv/JL/model/rwkv-0', strategy='cuda fp16')
#model ='rwkv'
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.0, top_k=0, # top_k = 0 then ignore
                     alpha_frequency = 0.0,
                     alpha_presence = 0.0,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [24], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
print('RWKV finish!!!')

########################################################################################################
from world.speech_encoder import SpeechEncoder

speech_encoder = SpeechEncoder(
    '/home/rwkv/JL/audio',
    768,
    downsample_K=5,
    hidden_dim=2048,
    train_mode="adapter",
    device='cuda',
)


import librosa
msg = 'Assistant:'

mod_path = '/home/rwkv/JL/RWKV-WORLD/demo/output5.wav'
audio, sample_rate = librosa.load(mod_path, sr=16000)  # sr=None 保持原采样率
y, _ = speech_encoder(audio)
result = pipeline.generate(msg, token_count=500, args=args, callback=None, state=None, sign=y)
print(result)
