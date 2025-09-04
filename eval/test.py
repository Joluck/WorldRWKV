from infer.worldmodel import Worldinfer
from PIL import Image


llm_path='/home/outmodel/new-1.5b/step3-0829/rwkv-0'
encoder_path='/home/model/siglip'
encoder_type='siglip'

model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

img_path = './docs/00f8281fc022933fd02609148710d8ae.png'
image = Image.open(img_path).convert('RGB')

text = '<|vision_end|>\x16User: 识别图中文字\x17\x16Assistant:'

result,_ = model.generate(text, image)

print(result)
