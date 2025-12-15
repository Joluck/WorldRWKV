import torch
from mlm.processing_mlm import ProcessMLM
from mlm.modeling_mlm import RWKV7VLForConditionalGeneration  

path = "/home/rwkv/jl/models/rwkv7-0.4b-8k"
model = RWKV7VLForConditionalGeneration.from_pretrained(path, trust_remote_code=True,torch_dtype=torch.bfloat16)
processor = ProcessMLM(path, trust_remote_code=True)

model = model.cuda()
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": './docs/03-Confusing-Pictures.jpg',
            },
            {"type": "text", "text": "Describe image"},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=None,messages=messages).to('cuda')
generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=1.0,
    top_p=0.0,
    repetition_penalty=1.0
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

response = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(response)
