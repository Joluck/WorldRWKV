# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from PIL import Image

# from transformers import AutoModel, SiglipImageProcessor
# from world.utils import read_and_merge_json

# class SiglipEncoder(nn.Module):
    
#     def __init__(
#         self,
#         encoder_path,
#         device="cuda",) -> None:
#         super(SiglipEncoder, self).__init__()

        
#         self.device = device
#         self.model = AutoModel.from_pretrained(encoder_path).vision_model
#         self.image_processor = SiglipImageProcessor.from_pretrained(encoder_path)
#         self.encoder_dim = 768  #self.model.config.hidden_size

#     def forward(self, x):

#         x= torch.from_numpy(self.image_processor(x)['pixel_values'][0]).to(self.device,dtype=torch.bfloat16)
#         x = self.model(x.unsqueeze(0), output_hidden_states=True).last_hidden_state
        
#         return x


# if __name__ == "__main__":
#     encoder_path = '/home/rwkv/models/siglip2'
#     data_file = '/home/rwkv/data/sharept/data'
#     model = SiglipEncoder(encoder_path).to(dtype=torch.bfloat16, device='cuda')
    
#     datas = read_and_merge_json(data_file)
#     print(len(datas))
#     for idx in range(len(datas)):
#         img_name = datas[idx]['image']
#         conversation_text = datas[idx]['conversations']

#         mod_path = f'{data_file}/{img_name}' 
#         image = Image.open(mod_path).convert('RGB')
#         vision_pixel = model(image)
#         print(vision_pixel.shape)
#         break
from datasets import  Features, Value, Image, Sequence
from datasets import Dataset as arrow_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, SiglipImageProcessor
import numpy as np
from tqdm import tqdm  # 进度条
from world.utils import read_and_merge_json, process_vision_token
class SiglipEncoder(nn.Module):
    def __init__(self, encoder_path, device="cuda"):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(encoder_path).vision_model.to( device='cuda')
        self.image_processor = SiglipImageProcessor.from_pretrained(encoder_path)
        self.encoder_dim = 768  # SigLip Base模型的隐藏层维度

    def forward(self, images):
        # 批量处理图片：image_processor自动返回batched tensor
        x= torch.from_numpy(self.image_processor(images)['pixel_values'][0]).to(self.device,dtype=torch.bfloat16)
        x = self.model(x.unsqueeze(0), output_hidden_states=True).last_hidden_state
        
        return x.squeeze(0)

class ImageDataset(Dataset):
    def __init__(self, datas, data_path):
        self.data_path = data_path
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):

            data = self.datas[idx]
            image_path = f"{self.data_path}/{data['image']}"
            image = Image.open(image_path).convert("RGB")
            conversations = data['conversations']
            text_tokens, text_labels = process_vision_token(conversations)
            return image, text_tokens, text_labels


def collate_fn(batch):
    image, text_tokens, text_labels = zip(*batch)
    
    image_batch = list(image)  
    
    text_token_batch = list(text_tokens)
    labels_token_batch = list(text_labels)


    return image_batch, text_token_batch, labels_token_batch

def main():
    # 配置参数
    encoder_path = "/home/rwkv/models/siglip2"
    data_file = "/home/rwkv/data/sharept/data"
    batch_size = 1  # 根据GPU显存调整
    num_workers = 1  # 数据加载线程数
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型（启用混合精度和编译优化）
    model = SiglipEncoder(encoder_path, device=device).to(dtype=torch.bfloat16)
    # model = torch.compile(model)  # PyTorch 2.0+ 加速

    # 加载数据路径
    datas = read_and_merge_json(data_file)
    
    # 创建DataLoader
    dataset = ImageDataset(datas, data_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,  # 加速数据到GPU的传输
    )

    # 批量推理
    features_list = []
    for batch_images, text_tokens, target_tokens in tqdm(dataloader, desc="Processing images"):
        for image, text_token, target_token in zip(batch_images, text_tokens, target_tokens):
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    image_features = model(image)  # 输入一个batch的PIL图像
            image_features = image_features.cpu().numpy().tolist()  # 转为Python列表
            text_token = text_token.cpu().numpy().tolist()  # 转为Python列表
            target_label = target_token.cpu().numpy().tolist()  # 转为Python列表
            sample = {'image_features':image_features, 'text_tokens': text_token, 'target_tokens':target_label}
            features_list.append(sample)

    features = Features({
        "image_features": Sequence(Sequence(Value("float32"))),  # 图像特征，通常是二维的
        "text_tokens": Sequence(Value("int64")),  # 文本token序列
        "target_tokens": Sequence(Value("int64")),  # 文本标签序列
    })



    # 3. 创建数据集
    dataset = arrow_dataset.from_list(features_list, features=features)

    out_file = '/home/rwkv/data/pt_arrow'
    dataset.save_to_disk(out_file,
                        max_shard_size="500MB",
                        num_proc=4,
                        )


if __name__ == "__main__":
    main()