from gen import Worldinfer
import librosa
import numpy as np
from datasets import load_dataset
import jsonlines
# # 打开并读取 JSON 文件
worldinfer = Worldinfer(model_path='/home/rwkv/JL/out_model/img-tt/rwkv-4', modality_path='/home/rwkv/JL/sdxl')

#with open(f'{args.data_file}/answer.jsonl', 'r') as file:
data_file='/home/rwkv/JL/audio-data/img-test'
with jsonlines.open(f'{data_file}/answer.jsonl') as file:
    data = list(file)
from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()  # 将图像转换为张量
])
for idx in range(4):
    mod_name = data[idx]['file_name']
    data_answer = data[idx]['answer']
    mod_path = f'{data_file}/{mod_name}'
    image = Image.open(mod_path).convert('RGB')
    sign = transform(image).to('cuda', torch.bfloat16)
    text = f'\n\nAssistant: {data_answer}'
    # res = worldinfer.prefill(text, sign)
    res = worldinfer.generate(sign)
    print('ori:  ', data_answer)
    print('res: ', res,'\n\n')

# from gen import Worldinfer
# import librosa
# import numpy as np
# from datasets import load_dataset
# import jsonlines

# # 打开并读取 JSON 文件
# #with open(f'{args.data_file}/answer.jsonl', 'r') as file:
# data_file='/home/rwkv/JL/audio-data/finish'
# with jsonlines.open(f'{data_file}/answer.jsonl') as file:
#     data = list(file)
# worldinfer = Worldinfer(model_path='/home/rwkv/JL/out_model/cnqa/rwkv-3')

# for idx in range(4):
#     mod_name = data[idx]['file_name']
#     data_answer = data[idx]['answer']
#     mod_path = f'{data_file}/{mod_name}'
#     audio, sample_rate = librosa.load(mod_path, sr=16000)  # sr=None 保持原采样率
#     #sign,_ = self.speech_encoder(audio)
#     sign = audio
#     res = worldinfer.generate(sign)
#     text = f'\x16Assistant: {data_answer}\x17'
#     # res = worldinfer.prefill(text, sign)
#     print('ori:  ', data_answer)
#     print('res: ', res)