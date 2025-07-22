
from torchvision import transforms
import torch
import io
import soundfile as sf
import numpy as np
from infer.rwkv.utils import PIPELINE
pipeline = PIPELINE('rwkv', "rwkv_vocab_v20230424")
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()  # 将图像转换为张量
])

def process_vision_token(conversations, max_length=2048, img_tokens=576):
    inputs = []
    labels = []

    replace = (f'<|vision_pad|>'*img_tokens+'<|vision_end|>')
    image_token = pipeline.encode(replace)
    image_label = [-100]*len(image_token)
    inputs += image_token
    labels += image_label

    for conv in conversations:
        role = conv.get('from', '').lower()
        content = conv.get('value', '')
        if role in ['user','human']:
            question = f"\x16User: {content}\x17"
            input = pipeline.encode(question)
            label = [-100]*len(input)
        elif role in ['assistant', 'gpt']:
            answer = f"\x16Assistant: {content}\x17"
            input = pipeline.encode(answer)
            label = input
        inputs += input
        labels += label

    inputs = torch.tensor(inputs, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    pad_length = max_length - len(labels) + 1
    final_input = F.pad(inputs, (0, pad_length), value=0)[:-1]
    final_label = F.pad(labels, (0, pad_length), value=-100)[1:]
    return final_input, final_label

def convert_vision_tensor(self, signs):
    signal = []

    for sign in signs:
        signal.append(self.modality(sign).squeeze(0))


    return signal


# def pad(inputs, labels, max_length):
#         text_inputs[i] = F.pad(token, (0, pad_len), value=0)[:-1]
#         text_labels[i] = F.pad(label, (0, pad_len), value=-100)[1:]



def process_tokens(conversations):

    # image_token = torch.tensor(pipeline.encode('<|vision_end|>'))
    # image_label = torch.full_like(image_token, -100)
    # inputs = [image_token]
    # labels = [image_label]
    inputs = []
    labels = []

    for conv in conversations:
        role = conv.get('from', '').lower()
        content = conv.get('value', '')
        
        if role in ['user','human']:
            question = f"\x16User: {content}\x17"
            input = torch.tensor(pipeline.encode(question))
            label = torch.full_like(input, -100)
        elif role in ['assistant', 'gpt']:
            answer = f"\x16Assistant: {content}\x17"
            input= torch.tensor(pipeline.encode(answer))
            label = input
        inputs.append(input)
        labels.append(label)
    inputs =torch.cat(inputs)
    labels =torch.cat(labels)
    return inputs, labels

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
    

import json
import os
from typing import List, Dict, Any

def read_and_merge_json(directory: str) -> List[Dict[str, Any]]:
    """
    读取目录下所有JSON文件并合并数据
    
    参数:
        directory: 要扫描的目录路径
        
    返回:
        合并后的JSON数据列表
    """
    merged_data = []
    
    # 确保目录存在
    if not os.path.isdir(directory):
        raise ValueError(f"目录不存在: {directory}")
    
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 如果数据是列表，则扩展merged_data
                    if isinstance(data, list):
                        merged_data.extend(data)
                    # 如果是字典，则追加到列表中
                    elif isinstance(data, dict):
                        merged_data.append(data)
                    else:
                        print(f"警告: 文件 {filename} 包含非字典/列表JSON数据，已跳过")
                        
            except json.JSONDecodeError:
                print(f"错误: 文件 {filename} 不是有效的JSON，已跳过")
            except Exception as e:
                print(f"错误: 处理文件 {filename} 时出错: {str(e)}")
    
    return merged_data


import json, jsonlines
import glob
from typing import List, Dict

def load_jsonl_files(file_pattern: str) -> List[Dict]:
    """
    读取匹配 file_pattern 的所有 JSONL 文件，并返回合并后的数据列表
    
    Args:
        file_pattern (str): 文件路径模式（支持通配符，如 `*.jsonl`）
    
    Returns:
        List[Dict]: 合并后的所有 JSON 数据
    """
    all_data = []
    for file_path in glob.glob(file_pattern):
        with jsonlines.open(file_path) as f:
            data = list(f)
            all_data+=data
    return all_data


import os
import glob

def load_vision_text(data_file):
    # 获取目录下所有文件的路径
    file_pattern = f'{data_file}/text/*'
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No files found in {file_pattern}")

    # 检查第一个文件的扩展名（假设目录下文件类型一致）
    first_file = files[0]
    _, ext = os.path.splitext(first_file)

    # 根据扩展名选择加载函数
    if ext == '.json':
        data = read_and_merge_json(f'{data_file}/text/*.json')
    elif ext == '.jsonl':
        data = load_jsonl_files(f'{data_file}/text/*.jsonl')
    else:
        raise ValueError(f"Unsupported file type: {ext}. Expected .json or .jsonl")

    return data



# 使用示例
# data_list = load_jsonl_files("/home/rwkv/data/vision_step2/text/*.jsonl")
# print(f"加载了 {len(data_list)} 条数据")
# # 使用示例
# if __name__ == "__main__":

#     try:
#         import time
#         start_time = time.time()

#         combined_data = read_and_merge_json("/home/rwkv/data/sharept/text")
#         end_time = time.time()
#         print(end_time-start_time)
#         print(f"成功合并了 {len(combined_data)} 条数据")
        

        
#     except Exception as e:
#         print(f"发生错误: {str(e)}")