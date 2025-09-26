import torch
from PIL import Image
from infer.worldmodel import Worldinfer

# =========================
# 模型路径配置
# =========================
llm_path = "/home/outmodel/test-step2/rwkv-0"
encoder_path = "/home/model/siglip"
encoder_type = 'siglip'

# =========================
# 初始化模型
# =========================
model = Worldinfer(
    model_path=llm_path,
    encoder_type=encoder_type,
    encoder_path=encoder_path
)

# =========================
# 加载图片
# =========================
img_path = '/home/peter/test/111.png'
image = Image.open(img_path).convert('RGB')

# =========================
# 构造 conversation 文本
# =========================
question = "请识别图中中文"
text_input = "<image>" + question

# =========================
# 调用模型生成回答
# =========================
result, _ = model.generate(text_input, image)

print("生成结果：\n", result)
