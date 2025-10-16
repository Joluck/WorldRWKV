
# from torchvision import transforms
import torch
import io
# import soundfile as sf
import numpy as np
from infer.rwkv.utils import PIPELINE
pipeline = PIPELINE('rwkv', "wr_vocab_v20230424")
import torch.nn.functional as F
import sys

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


def convert_texts_to_conversations(texts):
    
    conversations = []
    try:
        # 遍历texts中所有的user和assistant问答对
        for i, text_data in enumerate(texts):
            if isinstance(text_data, dict) and 'user' in text_data and 'assistant' in text_data:
                # 确保user和assistant字段不为空

                user_text = text_data.get("user", "").strip()
                assistant_text = text_data.get("assistant", "").strip()

                if user_text and assistant_text:
                    # 处理每一轮对话
                    conversations.append({'from': 'user', 'value': user_text})
                    conversations.append({'from': 'assistant', 'value': assistant_text})
        
        # 给第一个user的value添加<image>标记
        
        # 将conversations添加到sample中
        conversations
            
    except Exception as e:
        print(f"警告: 对话转换失败: {e}")
    return conversations

def process_vision_text(
    conversations, 
    tokenizer=None, 
    image_token_length=None,
    max_length=2048, 
    IGNORE_INDEX=-100,
    source=None,
    image_placeholder="<|placeholder|>"
):
    inputs = []
    labels = []
    index = 0
    for conv in conversations:
        role = conv.get('from', '').lower()
        content = conv.get('value', '')
        if role in ['user','human']:
            while image_placeholder in content:
                content = content.replace(image_placeholder, "<|image_pad|>" * image_token_length[index] , 1)
                index += 1
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
    import torch
    import torch.nn.functional as F
    import re
    # 模拟的 pipeline 编码器    
    class MockPipeline:
        def encode(self, text):
            # 模拟编码器，将文本转换为简单的数字列表
            return [ord(c) % 100 for c in text]

        # 模拟的 conversations 数据
        conversations = [
            {'from': 'user', 'value': '<image><image>\nThis is an  example <image>.'},
            {'from': 'assistant', 'value': 'This is the response.'},
            {'from': 'user', 'value': 'Another <image> example with <image> tokens.'},
            {'from': 'assistant', 'value': 'Another response.'}
        ]
        # 替换匹配到的内容为空字符串
        while conversations[0]['value'].startswith("<image>"):
            conversations[0]['value'] = conversations[0]['value'].replace("<image>", "", 1)
        # 模拟的 image_token_length 数据
        image_token_length = [3, 2, 4]

        # 调用函数
        final_input, final_label = process_vision_text(
            conversations,
            tokenizer=None,
            image_token_length=image_token_length,
            max_length=2048,
            IGNORE_INDEX=-100,
            source="test_source",
        )

        # 输出结果
        print("Final Input:", final_input)
        print("Final Label:", final_label)
