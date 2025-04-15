import torch
import torch.nn as nn
import torch.nn.functional as F

from TimeRWKV.TimeRWKV import Model


class TimeRWKVEncoder(nn.Module):
    
    def __init__(
        self,
        task_name = 'forecast',
        ckpt_path = '',
        patch_len = '16',
        d_model = 512,
        d_ff = 2048,
        e_layers = 6,
        n_heads = 8,
        dropout = 0.1,
        output_attention=False,
        factor=3,
        activation = "gelu"
        ):
        super(TimeRWKVEncoder, self).__init__()
        self.task_name = task_name
        self.ckpt_path = ckpt_path  # 如果有预训练模型路径，可以在这里指定
        self.patch_len = patch_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.output_attention = output_attention
        self.factor = factor
        self.activation = activation
        self.time_model = Model(configs=self)
    def forward(self, x):
        x = self.time_model(x,None,None,None,None)
        return x
    
if __name__ == "__main__":
    encoder = TimeRWKVEncoder(
        task_name='forecast',
        ckpt_path='',
        patch_len=16,
        d_model=512,
        d_ff=2048,
        e_layers=6,
        n_heads=8,
        dropout=0.1,
        output_attention=False,
        factor=3,
        activation="gelu"
    ).cuda()

    batch_size = 2
    sequence_length = 32
    feature_dim = 512
    input_data = torch.randn(batch_size, sequence_length, feature_dim).cuda()
    output = encoder(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")