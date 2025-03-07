
<h1 align="center">
  <p>WorldRWKV: Exploring RWKV7’s Understanding Capabilities of Any Modality in the World</p>
</h1>

\[ English | [中文](README_zh.md) \]
# 简介
我们的目标是用纯RWKV7架构实现任意模态训练推理；现在我们可以使用encoder来任意切换模态的输入并输出文本。未来逐步实现端到端的跨模态推理，并且使用RWKV7来探索World Model的雏形。目前项目处于初期阶段，仍然有很多地方需要优化，欢迎加入我们。
- 模型下载：[HFModel](https://huggingface.co/WorldRWKV).  
- 演示地址：[Demo](https://shoumenchougou.github.io/testforvideo/)
- 加入我们：[Discord](https://discord.com/invite/bDSBUMeFpc) QQ: 1015471226

## 发布
- [3/7] 🔥 发布仓库 **WorldRWKV: Exploring RWKV7’s Understanding Capabilities of Any Modality in the World**. 训练细节以及相关论文将在下周发布 [HFModel](https://huggingface.co/WorldRWKV).

# 环境
- 克隆仓库并进入文件
```
git clone https://github.com/JL-er/WorldRWKV.git
cd WorldRWKV
```
- 依赖
```
conda create -n world python=3.12
conda activate world
pip install -r requirements.txt #中国用户添加-i https://pypi.tuna.tsinghua.edu.cn/simple
# 推荐 torch=>2.4.0
```

# 推理
> [!NOTE]
> 请确保encoder model和encoder_type匹配. 更多细节在:world/world_encoder.py
```
from infer.worldmodel import Worldinfer
from PIL import Image


llm_path='/home/rwkv/model/rwkv7-3b-siglip/rwkv-0'
encoder_path='/home/rwkv/model/siglip2basep16s384'
encoder_type='siglip' #[clip, whisper, siglip, speech]

model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

img_path = './docs/03-Confusing-Pictures.jpg'
image = Image.open(img_path).convert('RGB')

text = '\x16User: What is unusual about this image?\x17Assistant:'

result = model.generate(text, image)

print(result)
```
## Web-demo (Using Gradio)
```
python audio_multiturns_web.py # For Audio QA and ASR
 
python visual_web.py  # For Visual QA 

```

# 训练
> [!NOTE]
> 请确保encoder model和encoder_type匹配，以及训练任务与data_type匹配。你也可以在world/world_encoder.py中注册自己的encoder类
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
| [**Clip**](https://huggingface.co/openai/clip-vit-large-patch14-336)    | RWKV7-0.4B     | 62.04      | 31.72      | 49.32       |   51.10         |
|| RWKV7-1.5B     | 72.31       | 40.27       | 54.56       |   62.77          |
|             | RWKV7-3B       | 73.13       | 45.56       | 57.00       | 70.06       |
| [**SigLIP2**](https://huggingface.co/google/siglip2-base-patch16-384) | RWKV7-0.4B     |    72.04     | 38.75       | 55.52       | 43.32       |
|             | RWKV7-1.5B     |     76.95    | 44.96       | 58.88       | 63.10       |
|             | RWKV7-3B       |     78.30     |   51.09          |   60.75          |     70.93        |

# 语音指标

| **Encoder** | **LLM** | **LibriSpeech** | **Aishell-1** |
|:--------------:|:--------------:|:--------------:|:--------------:|
|[**wavlm large**](https://huggingface.co/microsoft/wavlm-large) | RWKV7-0.4B | 2.51%(clean) | 9.68%(dev) |
|            |            | 7.72%(other) | 10.21%(test) |

## 语音识别 & 语音问答 (Demo)
| **Encoder** | **LLM** | **task** | **Checkpoint** |
|:--------------:|:--------------:|:--------------:|:--------------:|
|[**wavlm large**](https://huggingface.co/microsoft/wavlm-large) | RWKV7-0.1B | EN asr|[WorldRWKV/RWKV7-0.1B-wavlmLarge-ENASR-demo](https://huggingface.co/WorldRWKV/RWKV7-0.1B-wavlmLarge-ENASR-demo)|
|            |     RWKV7-0.4B       | EN asr|[WorldRWKV/RWKV7-0.4B-wavlmLarge-ENASR-demo](https://huggingface.co/WorldRWKV/RWKV7-0.4B-wavlmLarge-ENASR-demo)|
|            |     RWKV7-0.4B       | CN asr|[WorldRWKV/RWKV7-0.4B-wavlmLarge-CNASR-demo](https://huggingface.co/WorldRWKV/RWKV7-0.4B-wavlmLarge-CNASR-demo)|
|            |     RWKV7-0.4B       | EN qa|[WorldRWKV/RWKV7-0.4B-wavlmLarge-ENQA-demo](https://huggingface.co/WorldRWKV/RWKV7-0.4B-wavlmLarge-ENQA-demo)|
