import gradio as gr
import os
from datetime import datetime
import librosa
import numpy as np

from infer.worldmodel import Worldinfer

llm_path='/home/rwkv/JL/out_model/wavlm-mlp-0.1b-2/rwkv-0'
encoder_path='/home/rwkv/JL/audio'
encoder_type='speech'

model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)
def save_audio(audio):
    # 检查 audio 是否为 None
    if audio is None:
        return "未检测到音频，请重新录制。"

    # 解包元组
    sample_rate, audio_data = audio

    # 检查并转换音频数据为浮点数格式
    if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
        # 如果音频数据是整数格式，转换为浮点数并归一化
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    # 重采样到 16000 Hz
    resampled_audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    text = '\x16Assistant:'
    res,_ = model.generate(text, resampled_audio)
    return res

iface = gr.Interface(
    fn=save_audio,  # 处理录音的函数
    inputs=gr.Audio( type="numpy"),  # 录音输入
    outputs="text",  # 输出结果
    live=True,  # 实时处理
    title="WorldRWKV",
    description="点击录音按钮开始录音，录音结束后会自动保存。"
)

iface.launch(server_name="0.0.0.0")