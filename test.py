class Data():
    model_name_or_path: str = "RWKV/rwkv-5-world-3b"
    data_file :str = '/home/rwkv/JL/data/origin'
    ctx_len: int = 100
    micro_bsz: int = 2
    epoch_steps: int =1
    n_embd: int = 768
    data_type: str = 'origin'
import torch
def collate_fn(batch):
    # 解压 batch 中的数据
    signs, tokens = zip(*batch)
    
    # signs 保持 list of lists
    signs_batch = list(signs)  # 或直接使用 signs
    
    # 其他数据进行 stack
    tokens_batch = list(tokens)
    return signs_batch, tokens_batch
from torch.utils.data import DataLoader
from world.dataset import WorldDataset
if __name__ == "__main__":
    args = Data()
    data = WorldDataset(args)
    dataloader = DataLoader(
        dataset=data,
        batch_size=2,
        shuffle=False,  # 如果需要随机打乱数据，设为 True
        num_workers=0,  # 单进程加载
        pin_memory=True,
        collate_fn = collate_fn
    )
    for batch in dataloader:
        audio, token = batch
