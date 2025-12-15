import base64
import json
import io
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uvicorn
import argparse
from mlm.encoder.siglip_encoder import SiglipEncoder
import types
import torch
from mlm.processing_mlm import ProcessMLM
from mlm.modeling_mlm import RWKV7VLForConditionalGeneration  



parser = argparse.ArgumentParser(description="WorldRWKV API")
parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
parser.add_argument("--port", type=int, default=8000, help="监听端口")
parser.add_argument("--llm_path", type=str, required=True, help="rwkv 模型路径")

args = parser.parse_args()

current_state = None
first_question = False

# model = Worldinfer(model_path=args.llm_path, encoder_type=args.encoder_type, encoder_path=args.encoder_path)
model = RWKV7VLForConditionalGeneration.from_pretrained(args.llm_path, trust_remote_code=True,torch_dtype=torch.bfloat16).eval()
# tokenizer = AutoTokenizer.from_pretrained('/home/rwkv/JL/g1fla', trust_remote_code=True)
processor = ProcessMLM(args.llm_path, trust_remote_code=True)

model = model.cuda()


app = FastAPI(title="WorldRWKV API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageUrl(BaseModel):
    url: str

class TextContent(BaseModel):
    type: str = "text"
    text: str

class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: ImageUrl

class Message(BaseModel):
    role: str
    content: List[Union[TextContent, ImageContent]]|str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str

def base64_to_pil_image(base64_str: str) -> Image.Image:
    try:
        # 分离数据URI前缀
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # 解码并强制转换为RGB格式
        image_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_data))
        return img.convert('RGB')  # 确保输出总是3通道RGB
    except Exception as e:
        raise ValueError(f"图像处理失败: {str(e)}")

def process_messages(messages: List[Message]) -> tuple[str, Optional[Image.Image]]:
    global current_state, first_question

    text_content = ""
    image = None

    for message in messages:
        if message.role == "user":
            for content in message.content:
                if isinstance(content, TextContent):
                    text_content = content.text
                    if image is not None:
                        break
                elif isinstance(content, ImageContent):
                    if content.image_url.url.startswith('data:image'):
                        image = base64_to_pil_image(content.image_url.url)

    return text_content, image

def convert_messages(messages_obj):
    messages_dict = []
    for msg in messages_obj:
        images = []
        texts = []
        for item in msg.content:
            if isinstance(item, TextContent):
                texts.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                images.append({"type": "image", "image": item.image_url.url})
        # 图片放前，文本放后
        content_list = images + texts
        messages_dict.append({"role": msg.role, "content": content_list})
    return messages_dict


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    global current_state, first_question

    try:
        messages = convert_messages(request.messages)
        user_input, image = process_messages(request.messages)

        if not user_input:
            raise HTTPException(status_code=400, detail="No text content found in messages")

        if image is None:
            raise HTTPException(status_code=400, detail="Image is required")
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image] ).to('cuda')
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

        result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0].replace('\u0017', '')
        current_state = None
        bot_response = result

        final_response = bot_response

        import time
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_response
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(user_input.split()),
                "completion_tokens": len(final_response.split()),
                "total_tokens": len(user_input.split()) + len(final_response.split())
            }
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "dummy",
                "object": "model",
                "created": 1640995200,
                "owned_by": "world-rwkv"
            }
        ]
    }

@app.get("/")
async def root():
    return {"message": "WorldRWKV API Server", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug"
    )