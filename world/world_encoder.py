import torch
import torch.nn as nn
import torch.nn.functional as F
from world.encoder.speech_encoder import SpeechEncoder
# from world.encoder.visual_encoder import VisualEncoder
from world.encoder.whisper_encoder import WhisperEncoder
from world.encoder.clip_encoder import ClipEncoder
from world.encoder.siglip_encoder import SiglipEncoder


class WorldEncoder(nn.Module):
    def __init__(self, encoder_type: str, **kwargs):
        super().__init__()
        self.world_encoder = self._build_encoder(encoder_type, kwargs)
        
    def _build_encoder(self, encoder_type: str, config: dict):
        encoder_map = {
            "clip": ClipEncoder,
            "whisper": WhisperEncoder,
            # "visual": VisualEncoder,
            "speech": SpeechEncoder,
            "siglip": SiglipEncoder
        }
        
        if encoder_type not in encoder_map:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
            
        # 动态过滤有效参数
        encoder_class = encoder_map[encoder_type]
        
        return encoder_class(**config)
    
    def forward(self, x):
        return self.world_encoder(x)
    
    def load_checkpoint(self, state_dict):
        self.world_encoder.load_state_dict(state_dict, strict=False)