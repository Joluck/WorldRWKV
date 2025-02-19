import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration

class SpeechAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpeechAdapter, self).__init__()
        self.downsample_K = 10
        self.linear2linear = nn.Sequential(
            nn.Linear(input_dim * self.downsample_K, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim),
        )
    def forward(self, x):
        if x.size(1)<5 or x.size(1)>5000:
            return False
        # reshape the output from [batch_size, num_frames, hidden_size] to [batch_size, num_frames//downsample_K, hidden_size*downsample_K]
        x = x.unfold(1, self.downsample_K, self.downsample_K).flatten(2)
        x = self.linear2linear (x)
        return x



class WhisperEncoder(nn.Module):
    def __init__(
        self,
        encoder_path,
        project_dim,
        train_mode="adapter",
        device="cuda",
    ):
        assert train_mode in ["adapter", "full"]
        super(WhisperEncoder, self).__init__()
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(encoder_path)

        self.model = WhisperForConditionalGeneration.from_pretrained(encoder_path).model.encoder

        self.model_output_dim = self.model.config.d_model
            
        self.project_dim = project_dim
        self.adapter = SpeechAdapter(self.model_output_dim, self.project_dim)

    def forward(self, x):
        input_dict = self.processor(
            x, return_tensors="pt", sampling_rate=16000
        ).to(self.device,dtype=torch.bfloat16)
        
        # encoder only
        # x = self.model(**input_dict).last_hidden_state

        # stf encoder
        x = self.model(**input_dict).last_hidden_state
        
        x= self.adapter(x)#x:(B,T,hidden dim)
        # mask = torch.ones(x.shape[0],x.shape[1]).to(self.device,dtype=torch.bfloat16)
        return x
