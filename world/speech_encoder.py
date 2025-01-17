import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

from transformers import AutoProcessor, AutoModel
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer


class SpeechEncoder(nn.Module):
    def __init__(
        self,
        model_id,
        project_dim,
        downsample_K=5,
        hidden_dim=2048,
        train_mode="adapter",
        device="cuda",
    ):
        assert train_mode in ["adapter", "full"]
        super(SpeechEncoder, self).__init__()

        # feature_extractor = Wav2Vec2FeatureExtractor(
        #     feature_size=1,
        #     sampling_rate=16000,
        #     padding_value=0.0,
        #     do_normalize=True,
        #     return_attention_mask=False,
        # )
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.time_reduction_factor = int(
            self.processor.feature_extractor.sampling_rate / 50
        )
        self.padding_length = 320
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True)
        self.model_output_dim = self.model.config.hidden_size
        self.downsample_K = downsample_K
        self.project_dim = project_dim
        if hidden_dim is None:
            self.hidden_dim = self.project_dim * 2
        else:
            self.hidden_dim = hidden_dim
        #adapter shall be a Linear(Relu(Linear)) structure
        self.adapter = nn.Sequential(
            nn.Linear(self.model_output_dim * self.downsample_K, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )#.to(self.device,dtype=torch.bfloat16)
        #self.set_gradient(train_mode)


    def calculate_mask(self, input_dict):
        """
        Also need to handle the masking issue, to let the model not to attend to the padding tokens
        """
        attention_mask = input_dict["attention_mask"]  # [batch, num_samples]
        length_in_samples = (
            attention_mask.shape[1] // self.padding_length * self.padding_length
        )
        # calculate the mask length
        mask_length = length_in_samples // self.time_reduction_factor
        # create the mask
        mask = attention_mask[:, :: (self.time_reduction_factor * self.downsample_K)]
        return mask

    def forward(self, x):
        input_dict = self.processor(
            x, return_tensors="pt", padding=True, sampling_rate=16000
        ).to(self.device,dtype=torch.bfloat16)
        #mask = self.calculate_mask(input_dict)
        #x = self.model(input_dict['input_values'].to('cuda:0')).last_hidden_state.to(self.device)
        x = self.model(**input_dict).last_hidden_state

        if x.size(1)<5 or x.size(1)>5000:
            return False
        # reshape the output from [batch_size, num_frames, hidden_size] to [batch_size, num_frames//downsample_K, hidden_size*downsample_K]
        x = x.unfold(1, self.downsample_K, self.downsample_K).flatten(2)
        x = self.adapter(x)
        #mask = mask[:, : x.shape[1]]
        return x
    

# class SpeechAdapter(SpeechEncoder):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.adapter = nn.Sequential(
#             nn.Linear(self.model_output_dim * self.downsample_K, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.project_dim),
#         )
#     def forward(self, x):
#         return self.adapter(x)
    
# speech_encoder = SpeechEncoder(
#     '/home/rwkv/JL/audio',
#     768,
#     downsample_K=5,
#     hidden_dim=2048,
#     train_mode="adapter",
#     device='cuda',
# )

# 方法1：使用 pandas
# import pandas as pd
# import librosa
# import io
# import soundfile as sf
# # # 读取parquet文件
# # df = pd.read_parquet('/home/rwkv/JL/audio/test.parquet')
# def bytes_to_audio(audio_bytes):
#     with io.BytesIO(audio_bytes) as buf:
#         # 使用 soundfile 读取音频数据
#         audio_array, sr = sf.read(buf)
        
#         # 确保是单声道
#         if len(audio_array.shape) > 1:
#             audio_array = audio_array.mean(axis=1)
        
#         # 确保是 float32 类型
#         audio_array = audio_array.astype(np.float32)
        
#         return {
#             'array': audio_array,
#             'sampling_rate': sr
#         }

# audio_data0 = bytes_to_audio(df['question_audio'][0]['bytes'])
# audio0 = librosa.resample(audio_data0['array'],orig_sr= 48000,target_sr= 16000)

# audio_data1 = bytes_to_audio(df['question_audio'][1]['bytes'])
# audio1 = librosa.resample(audio_data1['array'],orig_sr= 48000,target_sr= 16000)
# list = [audio0, audio1]
# y, _ = speech_encoder(list)
# print(y.shape)