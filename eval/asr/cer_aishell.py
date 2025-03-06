from infer.worldmodel import Worldinfer

import librosa
from datasets import load_dataset

llm_path='/home/rwkv/JL/out_model/wavlm-mlp-0.1b/rwkv-0'
encoder_path='/home/rwkv/JL/audio'
encoder_type='speech'

model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)
# dataset = load_dataset('/home/rwkv/JL/data/fixie-ai-librispeech_asr/clean')
# data = load_dataset("JerryAGENDD/chinese_speech_cosy_audio", cache_dir = "../temp_datasets")['train']

# data = load_dataset("fixie-ai/librispeech_asr",'clean')['test'].shuffle()

data = dataset = load_dataset("carlot/AIShell")['test']
print(len(data))

# 初始化WER计算的变量
from jiwer import wer
from tqdm import tqdm
from jiwer import cer
# 初始化保存生成内容和正确答案的列表
generated_texts = []
reference_texts = []

# 使用tqdm进度条迭代数据集
with tqdm(total=len(data), desc="Processing", unit="sample") as pbar:
    for idx in range(len(data)):
        sample = data[idx]
        audio = sample['audio']
        #data_answer = sample['text'].lower()
        data_answer = sample['transcript'].replace(" ","")
        audio = librosa.resample(audio['array'], orig_sr=audio['sampling_rate'], target_sr=16000)
        
        # zeros = np.zeros(1600)
        # audio = np.concatenate((zeros, audio))
        text='\x16Assistant:'
        
        res,_ = model.generate(text,audio)
        res = res.lstrip()
        # 保存生成内容和正确答案 
        generated_texts.append(res)
        reference_texts.append(data_answer)
        tqdm.write(res)
        tqdm.write(data_answer)
        tqdm.write("\n")
        # 更新进度条
        pbar.update(1)

# 计算最终的总WER
# final_wer = wer(reference_texts, generated_texts)
final_wer = cer(reference_texts, generated_texts)
# 打印最终的总WER
print(f"CER: {final_wer:.4f}")