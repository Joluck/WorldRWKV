import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Sequence
from PIL import Image


from transformers import AutoModel, SiglipImageProcessor

class VisualAdapter(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, encoder_dim, project_dim, hidden_dim=None):

        super().__init__()
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim

        if self.hidden_dim==None:
            self.hidden_dim = project_dim*2

        self.pre_norm = nn.LayerNorm(self.project_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.encoder_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )
        # self.proj = nn.Sequential(
        #     nn.Linear(self.encoder_dim, self.hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_dim, self.project_dim),
        # )

    
    def forward(self, x):        
        x = self.proj(x)
        return x + self.pre_norm(x)



class SiglipEncoder(nn.Module):
    
    def __init__(
        self,
        encoder_path,
        project_dim,
        train_mode="adapter",
        device="cuda" if torch.cuda.is_available() else "cpu",) -> None:
        super(SiglipEncoder, self).__init__()

        
        self.device = device
        self.model = AutoModel.from_pretrained(encoder_path).vision_model.to(self.device)
        self.image_processor = SiglipImageProcessor.from_pretrained(encoder_path)
        self.encoder_dim = 768  #self.model.config.hidden_size
        
        self.adapter = VisualAdapter(self.encoder_dim, project_dim).to(self.device)
    def forward(self, images:Union[Image.Image, Sequence[Image.Image]]):
        """
        Encode single image or a list of images, preserving each image as a separate token
        
        Args:
            images: List of PIL images to encode
        
        Returns:
            Tensor of shape (num_images, project_dim) where each row represents an encoded image
        """
        # Process all images
        try:
            # Process images with the image processor
            if isinstance(images, Image.Image):
                images = [images]
                
            processed_images = [self.image_processor(img)['pixel_values'][0] for img in images]
            # Stack images and move to the correct device
            x = torch.tensor(np.stack(processed_images)).to(self.device)
            
            # Get features from vision model
            with torch.no_grad():
                x = self.model(x, output_hidden_states=True).last_hidden_state
            
            # Apply adapter to get projected features
            features = self.adapter(x)
            
            # Return features for all images (no pooling)
            return features
        except Exception as e:
            print(f"Error in SiglipEncoder forward pass: {e}")
            raise
