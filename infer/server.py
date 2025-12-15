import base64
import json
import io
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uvicorn
from infer.worldmodel import Worldinfer
import argparse
# llm_path = "/home/rwkv/models/0808test/step2/rwkv-0"
# encoder_path = "/home/rwkv/models/siglip2"
# encoder_type = 'siglip'
parser = argparse.ArgumentParser(description="WorldRWKV API")
parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
parser.add_argument("--port", type=int, default=8000, help="监听端口")
parser.add_argument("--llm_path", type=str, required=True, help="rwkv 模型路径")
parser.add_argument("--encoder_path", type=str, required=True, help="视觉 encoder 路径")
parser.add_argument("--encoder_type", type=str, default="siglip", help="encoder 类型")
args = parser.parse_args()

current_state = None
first_question = False
if 'auto' in args.encoder_type:
    processor = 'auto'
else:
    processor = None
model = Worldinfer(model_path=args.llm_path, encoder_type=args.encoder_type, encoder_path=args.encoder_path, processor=processor)

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

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    global current_state, first_question

    try:
        user_input, image = process_messages(request.messages)

        if not user_input:
            raise HTTPException(status_code=400, detail="No text content found in messages")

        if image is None:
            raise HTTPException(status_code=400, detail="Image is required")

        
        result, state = model.generate(user_input, image, state=None)
        current_state = state
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