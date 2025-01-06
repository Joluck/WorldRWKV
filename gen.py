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

from world.speech_encoder import SpeechEncoder


class Worldinfer():
    def __init__(self, model_path, modality_path='/home/rwkv/JL/audio', strategy='cuda fp16'):
        model = RWKV(model=model_path, strategy=strategy)
        self.pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


        self.args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.0, top_k=0, # top_k = 0 then ignore
                            alpha_frequency = 0.0,
                            alpha_presence = 0.0,
                            token_ban = [0], # ban the generation of some tokens
                            token_stop = [24], # stop generation whenever you see any token here
                            chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
        print('RWKV finish!!!')
        self.modality = SpeechEncoder(
            modality_path,
            768,
            downsample_K=5,
            hidden_dim=2048,
            train_mode="adapter",
            device='cuda',
        ).to('cuda', torch.bfloat16)


        self.msg = '\x16Assistant:'


    def generate(self, audio):
        y= self.modality(audio)
        result = self.pipeline.generate(self.msg, token_count=500, args=self.args, callback=None, state=None, sign=y)
        return result

# worldinfer = Worldinfer(model_path='/home/rwkv/JL/out_model/asr-step2/rwkv-0')

# import librosa
# msg = '\x16Assistant:'
# from datasets import load_dataset
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