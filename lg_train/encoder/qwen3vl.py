import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor



class Qwen3VLEncoder(nn.Module):
    
    def __init__(
        self,
        encoder_path,
        project_dim,
        train_mode="adapter",
        device="cuda",) -> None:
        super(Qwen3VLEncoder, self).__init__()

        
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(encoder_path).visual
        self.model.deepstack_merger_list = nn.ModuleList([])

        self.encoder_dim = 768  #self.model.config.hidden_size

        # self.adapter = VisualAdapter(self.encoder_dim, project_dim)
    def forward(self, x):
        x = self.model(x['pixel_values'], x['image_grid_thw'], output_hidden_states=True)
        return x


if __name__ == "__main__":
    from PIL import Image
    encoder_config = {
        'encoder_path': "/home/rwkv/models/qwen3-vl2b-inst/",
        'project_dim' : 768
    }

    qwen = Qwen3VLEncoder(**encoder_config)
    image1 = Image.open('/home/rwkv/data/vision_step2/data/chartqa/train/png/34.png').convert('RGB')
    image2 = Image.open('/home/rwkv/data/vision_step2/data/chartqa/train/png/43.png').convert('RGB')
    images = [image1]

    processor = AutoProcessor.from_pretrained("/home/rwkv/models/qwen3-vl2b-inst/")
    image_inputs = processor.image_processor(images=images)

    print(processor.image_processor.merge_size)
    image_grid_thw = image_inputs["image_grid_thw"]
    merge_length = processor.image_processor.merge_size**2
    for index in range(len(image_grid_thw)):
        num_image_tokens = image_grid_thw[index].prod() // merge_length
        print(num_image_tokens)
    print(image_inputs['pixel_values'].shape)
    y, x = qwen(image_inputs)


    print(y.shape, x)