import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration

class SpeechAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpeechAdapter, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=3072, kernel_size=3, stride=2)
        self.transformer = nn.TransformerEncoderLayer(d_model=3072, nhead=8, dim_feedforward=4096)
        self.linear = nn.Linear(3072, output_dim)
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        # x shape: (batch_size, input_dim, seq_len)
        x = self.conv(x)
        # x shape after conv: (batch_size, input_dim, new_seq_len)
        x = x.permute(2, 0, 1)  # Transformer expects (seq_len, batch_size, input_dim)
        # x = self.transformer(x, src_key_padding_mask=mask.bool())
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, input_dim)
        x = self.linear(x)
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
