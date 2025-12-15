from infer.worldmodel import Worldinfer
from PIL import Image


llm_path='/home/rwkv/jl/outmodel/mod-0.4b-encoder-auto/rwkv-0'
encoder_path='/home/rwkv/models/siglip2'
encoder_type='siglip2'
if 'auto' in encoder_type:
    processor = 'auto'
else:
    processor = None
model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path, processor=processor)

img_path = '/home/rwkv/jl/modality-linear-model/docs/03-Confusing-Pictures.jpg'
image = Image.open(img_path).convert('RGB')

text = 'describe image'

result,_ = model.generate(text, image)

print(result)