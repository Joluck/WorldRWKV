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

model = RWKV(model='/home/rwkv/JL/out_model/pretrain/rwkv-18', strategy='cuda fp16')
#model ='rwkv'
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.0, top_k=0, # top_k = 0 then ignore
                     alpha_frequency = 0.0,
                     alpha_presence = 0.0,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [24], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

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
import io
import soundfile as sf
# # 读取parquet文件
# df = pd.read_parquet('/home/rwkv/JL/audio/test.parquet')
def bytes_to_audio(audio_bytes):
    with io.BytesIO(audio_bytes) as buf:
        # 使用 soundfile 读取音频数据
        audio_array, sr = sf.read(buf)
        
        # 确保是单声道
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # 确保是 float32 类型
        audio_array = audio_array.astype(np.float32)
        
        return {
            'array': audio_array,
            'sampling_rate': sr
        }

msg = 'Assistant:'

# 方法1：使用 pandas
import pandas as pd
import librosa
import io
import soundfile as sf
# # 读取parquet文件

df = pd.read_parquet('/home/rwkv/JL/audio/test.parquet')
# for idx in range(7):
#     print(df['answer'][idx])
#     audio_data = bytes_to_audio(df['question_audio'][idx]['bytes'])
#     audio = librosa.resample(audio_data['array'],orig_sr= 48000,target_sr= 16000)
#     y, _ = speech_encoder(audio)
#     result = pipeline.generate(msg, token_count=500, args=args, callback=None, state=None, sign=y)
#     print(result)

mod_path = '/home/rwkv/JL/world/output2.wav'
audio, sample_rate = librosa.load(mod_path, sr=16000)  # sr=None 保持原采样率
y, _ = speech_encoder(audio)
result = pipeline.generate(msg, token_count=500, args=args, callback=None, state=None, sign=y)
print(result)
