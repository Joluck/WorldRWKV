# from infer.worldmodel import Worldinfer


# model = Worldinfer(model_path=xxxx, encoder_type='xxx', encoder_path=xxxx)

import torch
from infer.rwkv.utils import PIPELINE
pipeline = PIPELINE('rwkv', "rwkv_vocab_v20230424")
token = torch.tensor(pipeline.encode(f'<|xxx|>\n\n<|user|>\n\n<|assistant|>:\x17'))
print(token)