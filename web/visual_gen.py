from infer.worldmodel import Worldinfer
from PIL import Image


llm_path = "/home/rwkv/models/0808test/step2/rwkv-0"
encoder_path = "/home/rwkv/models/siglip2"
encoder_type = 'siglip'


model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

img_path = './docs/03-Confusing-Pictures.jpg'
image = Image.open(img_path).convert('RGB')

text = '\x16User: What is unusual about this image?\x17Assistant:'

result,_ = model.generate(text, image)

print(result)
