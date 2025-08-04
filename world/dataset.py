import torch
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import json, jsonlines
import pandas as pd
import librosa
import os
from .utils import pipeline, process_tokens, bytes_to_audio, process_vision_token, read_and_merge_json, load_jsonl_files, load_vision_text

import PIL.PngImagePlugin
# 增加MAX_TEXT_CHUNK的大小，默认是1MB，可以设置为更大的值，例如10MB
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024



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

class WorldDataset(Dataset):
    def __init__(self, args, emb=None):
        self.args = args
        self.index_manager = None
        self.emb = emb
        if args.data_type =='wav':

            # 打开并读取 JSON 文件
            #with open(f'{args.data_file}/answer.jsonl', 'r') as file:
            with jsonlines.open(f'{args.data_file}/answer.jsonl') as file:
                self.data = list(file)
        elif args.data_type=='img': 
            self.data = load_vision_text(args.data_file)
            print('datasets numbers:', len(self.data))
        elif args.data_type == 'arrow':
            from datasets import load_from_disk, concatenate_datasets, load_dataset
            import os
            
            # 获取主目录下所有子目录
            subdirs = [os.path.join(args.data_file, d) for d in os.listdir(args.data_file) 
                    if os.path.isdir(os.path.join(args.data_file, d))]
            
            if subdirs:
                # 加载每个子目录的数据集
                datasets = [load_from_disk(subdir) for subdir in subdirs]
                # 连接所有数据集
                self.data = concatenate_datasets(datasets)
                print(f"已连接{len(datasets)}个子目录的数据集，总大小: {len(self.data)}")
            else:
                # 如果没有子目录，直接加载主目录
                self.data = load_from_disk(args.data_file)
                print(f"从单一目录加载数据集，大小: {len(self.data)}")
        elif args.data_type =='hf' or args.data_type =='qa' or args.data_type =='cnqa' or args.data_type =='cnasr' or args.data_type =='tts':
            from datasets import load_dataset, concatenate_datasets

            def list_subdirectories(base_path):
                import os
                return [
                    name for name in os.listdir(base_path)
                    if os.path.isdir(os.path.join(base_path, name)) and not name.startswith('.')
                ]

            datasets = []
            files = list_subdirectories(args.data_file)
            if not files:
                datasets = load_dataset(args.data_file, split="train")
            else:
                for file in files:
                    dataset = load_dataset(f'{args.data_file}/{file}', split="train")
                    datasets.append(dataset)
                datasets = concatenate_datasets(datasets)
            self.data = datasets
            print(len(datasets))
            
        elif args.data_type == "jsonl":
            import jsonlines

            with jsonlines.open(args.data_file) as file:
                self.data = list(file)

        else:
            self.data = pd.read_parquet(args.data_file)

        

    def setup(self, rank, world_size, devices, shuffle):
        self.rank = rank
        self.world_size = world_size
        self.index_manager = GlobalIndexManager(rank=rank, device_num=devices, shuffle=shuffle)
    
    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz


    def __getitem__(self, idx):
        idx = self.index_manager.get_next_idx(idx_t=idx) if self.index_manager else idx
        args = self.args

        if args.data_type =='hf':
            sample = self.data[idx]
            audio = sample['audio']
            data_answer = sample['text'] #####caption
            audio = librosa.resample(audio['array'],orig_sr= audio['sampling_rate'],target_sr= 16000)  # sr=None 保持原采样率
            sign = audio
            text_tokens = torch.tensor(pipeline.encode(f'\x16Assistant: {data_answer}\x17'))
            text_labels = text_tokens
        elif args.data_type == 'img':
            img_name = self.data[idx]['image']
            conversation_text = self.data[idx]['conversations']
            mod_path = f'{args.data_file}/data/{img_name}' 
            sign = Image.open(mod_path).convert('RGB')
            text_tokens, text_labels = process_vision_token(conversation_text)
            

        elif args.data_type == 'arrow':
            sample = self.data[idx]
            image = sample['image']
            conversation_text = sample['conversations']
            sign = image.convert('RGB')
            text_tokens, text_labels = process_vision_token(conversation_text)
           
        return sign, text_tokens, text_labels
