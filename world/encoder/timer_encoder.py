import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM

class TimerEncoder(nn.Module):
    def __init__(
        self,
        prediction_length,
        device="cuda",
    ):
        super(TimerEncoder, self).__init__()
        self.device = device
        self.prediction_length = prediction_length

        self.model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)

            

    def forward(self, x):
        x = self.model.generate(x, max_new_tokens=self.prediction_length)
        
        return x


if __name__ == "__main__":
    encoder = TimerEncoder(prediction_length=96)
    batch_size, lookback_length = 1, 2880
    x = torch.randn(batch_size, lookback_length)
    x = encoder(x)
    print(x.shape)
    print("timer encoder test pass")
