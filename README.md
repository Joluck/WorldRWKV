
<h1 align="center">
  <p>WorldRWKV</p>
</h1>

\[ English | [中文](README_zh.md) \]
# 简介
# 环境
```
conda create -n world python=3.12
conda activate world
pip install -r requirements.txt
# 推荐 torch=>2.4.0
```
# 训练
```
load_model=/home/rwkvos/model/rwkv/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth
proj_dir=/home/rwkvos/peter/out_model/rwkv7-3b-pretrain-siglip
data_file=/home/rwkvos/data/hf-imgs/pretrain595

n_layer=32
n_embd=2560

encoder_path="google/siglip2-base-patch16-384" #选择你需要的encoder
encoder_type=siglip #在worldencoder中注册类型
data_type=hf_img #数据类型

micro_bsz=32
epoch_save=1
epoch_steps=18605 
ctx_len=2048


HF_ENDPOINT="https://hf-mirror.com" python world_train.py \   # 中国用户使用"https://hf-mirror.com"下载模型
--load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--data_type $data_type \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 1e-3 --lr_final 0 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--encoder_path $encoder_path --encoder_type $encoder_type \
--my_testing "x070" --train_step adapter rwkv #train_step 选择你要训练的部分，encoder、adapter、rwkv
```

# 推理
```
from infer.worldmodel import Worldinfer

assert encoder_type in [clip, whisper, siglip, speech, visual]

model = Worldinfer(model_path=xxxx, encoder_type='xxx', encoder_path=xxxx)
text = ''
mod = image/audio
result = model.generate(text, mod)
print(result)
```
# 功能
### WorldRWKV已实现的功能以及后续添加的功能
| Function      | Work |
|:--------------:|:-----------:|
| asr            | ✅          |
| speech to text | ✅          |
| visual to text | ✅          |
| text to speech | ❌          |
| text to visual | ❌          |
|speech to speech| ❌          |


# 视觉指标

| **Encoder** | **LLM** | **VQAV2** | **TextVQA** | **GQA** | **ScienceQA** |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| [**Clip**](https://huggingface.co/openai/clip-vit-large-patch14-336)    | RWKV7-1.5B     | 72.31       | 40.27       | 54.56       |   62.77          |
|             | RWKV7-3B       | 73.13       | 45.56       | 57.00       | 70.06       |
| [**SigLIP2**](https://huggingface.co/google/siglip2-base-patch16-384) | RWKV7-0.4B     |             | 38.75       | 55.52       | 43.32       |
|             | RWKV7-1.5B     |             | 44.96       | 58.88       | 63.10       |
|             | RWKV7-3B       |             |   51.09          |   60.75          |     70.93        |

