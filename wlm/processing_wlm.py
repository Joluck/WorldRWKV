from transformers import AutoTokenizer
from PIL import Image

class ProcessWLM:
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, pretrained_model_name_or_path, **kwargs):
        # 只加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        self.image_token = "<|vision_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

    def __getattr__(self, name):
        # 让 ProcessWLM 可以调用 tokenizer 的所有方法
        return getattr(self.tokenizer, name)

    def __call__(self, text, images, **kwargs):
        """
        对输入文本做处理，同时提取 messages 中的所有 image
        """
        # 编码文本
        replacement = (
            "<|vision_start|>"
            + f"<|image_pad|>" * 576
            + "<|vision_end|>"
        )

        text = text.replace("<image>", replacement)
        inputs = self.tokenizer([text], return_tensors="pt", **kwargs)

        inputs["mod_values"] = images
        return inputs
    def process_images(self, messages):
        images = []
        for msg in messages:
            for content in msg.get("content", []):
                if content.get("type") == "image":
                    path = content["image"]
                    img = Image.open(path).convert("RGB")
                    images.append(img)
        return images

# 使用示例
if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/rwkv/JL/image1.jpg"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    processor = ProcessWLM("/home/rwkv/JL/g1fla")
    inputs = processor(messages)
    print(inputs)
