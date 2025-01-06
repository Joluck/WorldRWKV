########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import torch.nn.functional as F

import json
import math
import random
import os
import sys
import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info
from rwkv.utils import PIPELINE
import time
pipeline = PIPELINE('rwkv6', "rwkv_vocab_v20230424")

import pandas as pd
import librosa
import io
import soundfile as sf
# 读取parquet文件

from .speech_encoder import SpeechEncoder

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



def get_data_by_l_version(trainer: L.Trainer, args):
    if L.__version__[0] == '2':
        train_data = MyDataModule(args)
    else:
        raise ValueError(f"Unsupported PyTorch Lightning version: {L.__version__}")
    return train_data

class GlobalIndexManager:
    def __init__(self, rank=0, device_num=1, shuffle=True):
        self.current_idx = 0
        self.rank = rank
        self.device_num = device_num
        self.shuffle = shuffle
        
    def get_next_idx(self, idx_t):
        if self.shuffle:
            idx = idx_t
        else:
            idx = self.current_idx * self.device_num + self.rank 
            self.current_idx += 1
        return idx

class MyDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_data = None
        
        
    def setup(self, stage=None):
        self.train_data = MyDataset(self.args)
        self.args.vocab_size = self.train_data.vocab_size
        self.train_data.real_epoch = self.trainer.current_epoch
        self.train_data.rank = self.trainer.global_rank
        self.train_data.world_size = self.trainer.world_size
        self.train_data.setup(self.trainer.global_rank, self.trainer.world_size, 
                              int(self.args.devices), self.args.data_shuffle)
        
    def train_dataloader(self):
        # must set shuffle=False, persistent_workers=False (because worker is in another thread)
        return DataLoader(
            self.train_data,
            shuffle=self.args.data_shuffle,
            pin_memory=True,
            batch_size=self.args.micro_bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True
        )

class WorldDataset(Dataset):
    def __init__(self, args, emb=None):
        self.args = args
        self.rank = 0
        self.real_epoch = 0
        self.world_size = 0
        self.index_manager = None
        self.emb = emb

        if args.data_type =='wav':
            import jsonlines

            # 打开并读取 JSON 文件
            #with open(f'{args.data_file}/answer.jsonl', 'r') as file:
            with jsonlines.open(f'{args.data_file}/answer.jsonl') as file:
                self.data = list(file)
                
        elif args.data_type =='hf' or args.data_type =='qa':
            from datasets import load_dataset
            dataset = load_dataset(args.data_file, split="train[:460000]")
            self.data = dataset
            print(len(dataset))
            # print(dataset['train'][0])
            # self.data = dataset["ENGLISH"]

        else:
            self.data = pd.read_parquet(args.data_file)

        # self.speech_encoder = SpeechEncoder(
        #     '/home/rwkv/JL/audio',
        #     args.n_embd,
        #     downsample_K=5,
        #     hidden_dim=2048,
        #     train_mode="adapter",
        #     device='cuda',
        # )
        

    def setup(self, rank, world_size, devices, shuffle):
        self.rank = rank
        self.world_size = world_size
        self.index_manager = GlobalIndexManager(rank=rank, device_num=devices, shuffle=shuffle)
    
    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz


    def __getitem__(self, idx):
        idx = self.index_manager.get_next_idx(idx_t=idx) if self.index_manager else idx
        args = self.args
        if args.data_type =='wav':

            mod_name = self.data[idx]['file_name']
            data_answer = self.data[idx]['answer']
            mod_path = f'{args.data_file}/{mod_name}'
            audio, sample_rate = librosa.load(mod_path, sr=16000)  # sr=None 保持原采样率
            #sign,_ = self.speech_encoder(audio)
            sign = audio
            token = torch.tensor(pipeline.encode(f'\x16Assistant: {data_answer}\x17'))
        elif args.data_type =='hf':
            sample = self.data[idx]
            audio = sample['audio']
            data_answer = sample['text'] #####caption
            audio = librosa.resample(audio['array'],orig_sr= audio['sampling_rate'],target_sr= 16000)  # sr=None 保持原采样率
            #sign,_ = self.speech_encoder(audio)
            sign = audio
            token = torch.tensor(pipeline.encode(f'\x16Assistant: {data_answer}\x17'))
        elif args.data_type =='qa':
            sample = self.data[idx]
            # audio = sample['speech_cosy'][0]
            # data_answer = sample['answer'] 

            audio = sample['question_audio']
            data_answer = sample['answer'] #####caption
            sign = librosa.resample(audio['array'],orig_sr= audio['sampling_rate'],target_sr= 16000)  # sr=None 保持原采样率

            token = torch.tensor(pipeline.encode(f'\x16Assistant: {data_answer}\x17'))
        else:
            data_audio = bytes_to_audio(self.data['question_audio'][idx]['bytes'])
            data_answer = self.data['answer'][idx]
            audio = librosa.resample(data_audio['array'],orig_sr= 48000,target_sr= 16000)
            #sign,_ = self.speech_encoder(audio)
            sign = audio
            token = torch.tensor(pipeline.encode(f'\x16Assistant: {data_answer}\x17'))
        #print(idx, f'Assistant: {data_answer}\x17')
        return sign, token
    
        