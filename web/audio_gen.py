from infer.worldmodel import Worldinfer
import librosa
import numpy as np
import soundfile as sf

# 模型路径
llm_path = '/home/rwkv/model/rwkv7-0.4b-cnasr-step2'
encoder_path = '/home/rwkv/model/facebookhubert-large-ls960-ft'
encoder_type = 'speech'

# 初始化模型
model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

# 加载音频文件
audio_path = './test_audio.wav'
audio_data, sample_rate = sf.read(audio_path)

# 确保音频是单声道
if len(audio_data.shape) > 1:
    audio_data = audio_data[:, 0]

# 检查并转换音频数据为浮点数格式
if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

# 重采样到 16000 Hz
resampled_audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

# 构造提示文本
text = '\x16Assistant:'

# 生成结果
result, _ = model.generate(text, resampled_audio)

print(result) 