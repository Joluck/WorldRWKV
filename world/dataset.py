import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from PIL import Image
import jsonlines
import librosa
from .utils import *

import PIL.PngImagePlugin
# 增加MAX_TEXT_CHUNK的大小，默认是1MB，可以设置为更大的值，例如10MB
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024

class WorldDataset(Dataset):
    def __init__(self, args, processor=None):
        """
        通用多模态数据集：
        支持 data_type = ['hf', 'img', 'arrow', 'jsonl', 'wav', 'state']
        """
        self.args = args
        self.processor = processor
        self.data_type = args.data_type

        # --- 1. 加载数据 ---
        if args.data_type == 'hf':
            self.data = self._load_hf_dataset(args.data_file)
        elif args.data_type == 'arrow':
            self.data = self._load_arrow_dataset(args.data_file)
        elif args.data_type in ['img', 'state']:
            self.data = self._load_vision_text(args.data_file)
            if hasattr(args, "copy"):
                self.data = self.data * args.copy
        elif args.data_type == 'jsonl':
            with jsonlines.open(args.data_file) as f:
                self.data = list(f)
        elif args.data_type == 'wav':
            with jsonlines.open(f'{args.data_file}/answer.jsonl') as f:
                self.data = list(f)
        else:
            raise ValueError(f"Unsupported data_type: {args.data_type}")
        data_nums = len(self.data)
        print(f"Loaded {len(self.data)} samples for {args.data_type} dataset.")
        if args.epoch_steps < data_nums :
            self.data = self.data.select(range(args.epoch_steps))
            print(f"Trimmed to {len(self.data)} samples for epoch_steps {args.epoch_steps}.")
    # ------------------------------
    # 数据加载函数
    # ------------------------------

    def _load_hf_dataset(self, path):
        """加载 Hugging Face 格式数据"""
        subdirs = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        datasets = []
        for subdir in subdirs:
            try:
                ds = load_dataset(subdir, split="train")
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ 跳过无效数据目录: {subdir}, 原因: {e}")

        if datasets:
            return concatenate_datasets(datasets)
        else:
            # 说明当前目录本身是dataset根目录
            return load_dataset(path, split="train")

    def _load_arrow_dataset(self, path):
        """加载 Arrow 格式（支持多个子目录）"""
        subdirs = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        if subdirs:
            datasets = [load_from_disk(sd) for sd in subdirs]
            return concatenate_datasets(datasets)
        return load_from_disk(path)

    def _load_vision_text(self, path):
        """可根据项目自定义 load_vision_text"""
        # 假设格式 [{"image": "xxx.jpg", "conversations": [...]}, ...]
        with jsonlines.open(os.path.join(path, "data.jsonl")) as f:
            return list(f)

    # ------------------------------
    # Dataset 必须方法
    # ------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        t = self.data_type

        if t == 'img':
            return self._process_img(sample)
        elif t == 'arrow':
            return self._process_arrow(sample)
        elif t == 'hf':
            return self._process_hf(sample)
        elif t == 'wav':
            return self._process_wav(sample)
        elif t == 'jsonl':
            return sample
        else:
            raise ValueError(f"Unsupported data_type in __getitem__: {t}")

    # ------------------------------
    # 各类型处理函数
    # ------------------------------

    def _process_img(self, sample):
        images = sample['image']
        if not isinstance(images, list):
            images = [images]
        images = [Image.open(os.path.join(self.args.data_file, "data", img)).convert("RGB") for img in images]

        conversations = sample["conversations"]
        conversations[0]["value"] = "<image>" + conversations[0]["value"]

        input_ids, label_ids = process_vision_text(conversations, max_length=self.args.ctx_len, image_token_length=[576]*len(images))
        return {"images": images, "input_ids": input_ids, "labels": label_ids}

    def _process_arrow(self, sample):
        images = [img.convert("RGB") for img in sample["images"]][:3]  # ctx_len limit 3 image
        texts = convert_texts_to_conversations(sample["texts"])
        source = sample['source']
        while texts[0]["value"].startswith("<image>"):
            texts[0]["value"] = texts[0]["value"].replace("<image>", "", 1)
        for i in range(len(images)):
                texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=[576]*len(images), source=source)
        return  images, input_ids, label_ids
    def _process_hf(self, sample):
        image = sample['image']
        images=[]
        if not isinstance(image, list) and image is not None:
            images = [image]
        images = [img.convert("RGB") for img in images]
        texts = sample['conversations']
        while texts[0]["value"].startswith("<image>"):
            texts[0]["value"] = texts[0]["value"].replace("<image>", "", 1)
        for i in range(len(images)):
                texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
        
        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=[576]*len(images))
        return  images, input_ids, label_ids

    def _process_wav(self, sample):
        audio = librosa.load(sample["path"], sr=16000)[0]
        return {"audio": audio, "text": sample.get("text", "")}


import lightning as L
from torch.utils.data import DataLoader

class WorldDataModule(L.LightningDataModule):
    def __init__(self, args, processor=None):
        super().__init__()
        self.args = args
        self.processor = processor

    def setup(self, stage=None):
        self.train_dataset = WorldDataset(self.args, self.processor)


    def train_dataloader(self):
        def custom_collate_fn(batch):
            signs, inputs_ids, labels = zip(*batch)
            all_images = list(signs)
            inputs_ids = torch.stack(inputs_ids, dim=0)
            labels = torch.stack(labels, dim=0)

            return all_images, inputs_ids, labels
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.micro_bsz,
            shuffle=True,    # Lightning 自动替换成 DistributedSampler
            collate_fn=custom_collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
