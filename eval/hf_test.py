from wlm.encoder.siglip_encoder import SiglipEncoder
import types
import torch
from wlm.processing_wlm import ProcessWLM
from wlm.modeling_rwkv7_wlm import RWKV7VLForConditionalGeneration  
model = RWKV7VLForConditionalGeneration.from_pretrained('/home/rwkv/JL/g1fla', trust_remote_code=True,torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained('/home/rwkv/JL/g1fla', trust_remote_code=True)
processor = ProcessWLM('/home/rwkv/JL/g1fla', trust_remote_code=True)

model = model.cuda()
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/rwkv/JL/03-Confusing-Pictures.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images = processor.process_images(messages)
print(images)
inputs = processor(text=text, images=images).to('cuda')
generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=1.0,
    top_p=0.3,
    repetition_penalty=1.2
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

response = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(response)
