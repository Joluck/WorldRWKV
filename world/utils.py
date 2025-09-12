
# from torchvision import transforms
import torch
import io
# import soundfile as sf
import numpy as np
from infer.rwkv.utils import PIPELINE
pipeline = PIPELINE('rwkv', "wr_vocab_v20230424")
import torch.nn.functional as F

# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor()  # 将图像转换为张量
# ])
def check_vision_token(conversations):
    for conv in conversations:
        role = conv.get('from', '').lower()
        content = conv.get('value', '')
        if role in ['user','human']:
            question = f"\x16User: {content}\x17"
        elif role in ['assistant', 'gpt']:
            answer = f"\x16Assistant: {content}\x17"    
    return question, answer




def process_vision_text(
    conversations, 
    tokenizer=None, 
    image_token_length=None,
    max_length=2048, 
    IGNORE_INDEX=-100,
):
    inputs = []
    labels = []
    visual_replicate_index_image = 0
    for conv in conversations:
        role = conv.get('from', '').lower()
        content = conv.get('value', '')
        if role in ['user','human']:
            image = ''
            if "<image>" in content:
                parts = content.split("<image>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>"
                        + f"<|image_pad|>" * image_token_length[visual_replicate_index_image]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                    visual_replicate_index_image += 1
                new_parts.append(parts[-1])
                content = "".join(new_parts)
            question = f"\x16User:{content}\x17"
            input = pipeline.encode(question)
            label = [IGNORE_INDEX]*len(input)
        elif role in ['assistant', 'gpt']:
            answer = f"\x16Assistant:{content}\x17"
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
        data = read_and_merge_json(f'{data_file}/text')
    elif ext == '.jsonl':
        data = load_jsonl_files(f'{data_file}/text/*.jsonl')
    else:
        raise ValueError(f"Unsupported file type: {ext}. Expected .json or .jsonl")

    return data
if __name__ == "__main__":
    import jsonlines
    file_path = '/home/rwkv/data/vision_step2/text/chartqa.jsonl'
    with jsonlines.open(file_path) as f:
        data = list(f)

    print(data[0])
    conversations = data[0]['conversations']
    it, ot = process_vision_text(conversations, image_token_length=[8,8])
    print(it)
    import torch

    B, L, H = 2, 7, 768         # batch=2, seq_len=10, hidden=768
    device = 'cpu'
    input_ids = torch.tensor([[2,2,2,1,1,1,1],[1,2,2,3,1,1,1]], dtype=torch.long)
    # 1.1 文本侧
    inputs_embeds = torch.zeros(B, L, H, device=device)      # [2, 10, 768]

    image_embeds = torch.randn(5, H, device=device) 

    image_mask = get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
    print('smask',image_mask, image_mask.shape)

                      # [2,10] 里 6 个 True

        # 2.1 把图像向量插到文本向量里
    inputs_embeds_after = inputs_embeds.masked_scatter(image_mask, image_embeds)


    print('inputs_embeds before :', inputs_embeds)     # [2, 10, 768]
    print('inputs_embeds after  :', inputs_embeds_after)  # [2, 10, 768]
