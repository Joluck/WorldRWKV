import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Sequence
from PIL import Image

from framefusion.main import FrameFusion
from framefusion.utils import TEXT_TOKEN, IGNORE_TOKEN

class SiglipFrameFusion(nn.Module):
    """
    Adapter to apply FrameFusion to SIGLIP encoder outputs.
    This reduces the number of tokens in video frame sequences by merging similar frames
    and pruning less important ones.
    """
    def __init__(self, siglip_encoder, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1):
        super().__init__()
        self.siglip_encoder = siglip_encoder
        self.framefusion = FrameFusion(cost, similarity_lower_bound, ratio_lower_bound)
        self.original_forward = siglip_encoder.forward
        
        # Replace the forward method
        self.siglip_encoder.forward = self.forward_with_framefusion

    def forward_with_framefusion(self, images: Union[Image.Image, Sequence[Image.Image]]):
        """
        Apply FrameFusion to SIGLIP encoder output to reduce the number of frame tokens.
        
        Args:
            images: Single image or list of PIL images to encode
            
        Returns:
            Tensor with reduced number of tokens based on FrameFusion algorithm
        """
        # Handle single image case (no need for FrameFusion)
        if isinstance(images, Image.Image) or len(images) <= 1:
            return self.original_forward(images)
        
        # Process images with the image processor
        processed_images = [self.siglip_encoder.image_processor(img)['pixel_values'][0] for img in images]
        # Stack images and move to the correct device
        x = torch.tensor(np.stack(processed_images)).to(self.siglip_encoder.device)
        
        # Get features from vision model without adapter
        with torch.no_grad():
            # Get the raw features from the vision model
            x = self.siglip_encoder.model(x, output_hidden_states=True).last_hidden_state
        
        # Now we have features of shape [num_frames, patches_per_frame, hidden_size]
        num_frames, patches_per_frame, hidden_dim = x.shape
        device = x.device
        total_patches = num_frames * patches_per_frame
        
        # Create a batch size of 1 for FrameFusion
        # Reshape to [1, num_frames*patches_per_frame, hidden_dim]
        features = x.reshape(1, total_patches, hidden_dim)
        
        # Create patch type tensor - each frame's patches get the same patch type (0 to num_frames-1)
        patch_type = torch.zeros(1, total_patches, dtype=torch.long, device=device)
        for i in range(num_frames):
            start_idx = i * patches_per_frame
            end_idx = (i + 1) * patches_per_frame
            patch_type[0, start_idx:end_idx] = i
        
        # Setup FrameFusion parameters
        self.framefusion.prepare(
            patch_type=patch_type,
            patch_num=num_frames,  # Number of different patch types (frames)
            image_token_start_index=torch.tensor([0], device=device),
            image_token_end_index=torch.tensor([total_patches], device=device),
            image_token_length=total_patches,
            original_length=total_patches,
            finish_merging=False,
            finish_pruning=False,
            sparsity_list=[]
        )
        
        # Create dummy position embeddings and attention mask for FrameFusion
        position_embeddings = [torch.zeros_like(features), torch.zeros_like(features)]
        attention_mask = None
        
        # Apply FrameFusion
        reduced_features, _, _ = self.framefusion(
            hidden_states=features,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )
        
        # Get the number of frames after reduction
        reduced_tokens = reduced_features.shape[1]
        
        # Calculate how many frames we have after reduction (rounded up)
        reduced_frames = (reduced_tokens + patches_per_frame - 1) // patches_per_frame
        
        # Create a new tensor with the right shape [reduced_frames, patches_per_frame, hidden_dim]
        # First, pad the reduced features to be a multiple of patches_per_frame
        padding_needed = (patches_per_frame - (reduced_tokens % patches_per_frame)) % patches_per_frame
        if padding_needed > 0:
            padding = torch.zeros(1, padding_needed, hidden_dim, device=device)
            reduced_features = torch.cat([reduced_features, padding], dim=1)
            reduced_tokens += padding_needed
        
        # Now reshape to match the expected output format
        final_features = reduced_features.reshape(reduced_frames, patches_per_frame, hidden_dim)
        
        # Apply the adapter to the final features
        final_features = self.siglip_encoder.adapter(final_features)
        
        # Log reduction statistics
        reduction_percentage = (1 - reduced_frames / num_frames) * 100
        print(f'FrameFusion reduced frames from {num_frames} to {reduced_frames} ({reduction_percentage:.2f}% reduction)')
        
        return final_features


class SiglipImageFrameFusion(nn.Module):
    """
    Adapter to apply FrameFusion to SIGLIP encoder outputs for single images.
    This reduces the number of patches within a single image by merging similar patches
    and pruning less important ones.
    """
    def __init__(self, siglip_encoder, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1):
        super().__init__()
        self.siglip_encoder = siglip_encoder
        self.framefusion = FrameFusion(cost, similarity_lower_bound, ratio_lower_bound)
        self.original_forward = siglip_encoder.forward
        
        # Replace the forward method
        self.siglip_encoder.forward = self.forward_with_framefusion

    def forward_with_framefusion(self, images: Union[Image.Image, Sequence[Image.Image]]):
        """
        Apply FrameFusion to SIGLIP encoder output to reduce the number of patches within an image.
        
        Args:
            images: Single image or list of PIL images to encode
            
        Returns:
            Tensor with reduced number of patches based on FrameFusion algorithm
        """
        # Convert single image to list if needed
        if isinstance(images, Image.Image):
            images = [images]
            
        # Process images with the image processor
        processed_images = [self.siglip_encoder.image_processor(img)['pixel_values'][0] for img in images]
        # Stack images and move to the correct device
        x = torch.tensor(np.stack(processed_images)).to(self.siglip_encoder.device)
        
        # Get features from vision model without adapter
        with torch.no_grad():
            # Get the raw features from the vision model
            x = self.siglip_encoder.model(x, output_hidden_states=True).last_hidden_state
        
        # For each image, apply FrameFusion to its patches
        results = []
        for i in range(len(images)):
            # Get the features for this image
            img_features = x[i:i+1]  # Shape: [1, patches_per_image, hidden_dim]
            num_patches = img_features.shape[1]
            hidden_dim = img_features.shape[2]
            device = img_features.device
            
            # Create patch types - for a single image, we'll group patches by their position
            # Group patches into a grid (e.g., 24x24 for 576 patches)
            grid_size = int(num_patches ** 0.5)  # Assuming square grid of patches
            
            # Create patch type tensor - assign types based on position in the grid
            patch_type = torch.zeros(1, num_patches, dtype=torch.long, device=device)
            for p in range(num_patches):
                # Assign patch types based on regions in the image (e.g., 3x3 grid of regions)
                row = p // grid_size
                col = p % grid_size
                region_size = max(1, grid_size // 3)  # Split into approximately 9 regions
                region_row = row // region_size
                region_col = col // region_size
                region_id = region_row * 3 + region_col  # Assign region ID (0-8 for a 3x3 grid)
                patch_type[0, p] = region_id
            
            # Setup FrameFusion parameters
            num_regions = 9  # 3x3 grid of regions
            self.framefusion.prepare(
                patch_type=patch_type,
                patch_num=num_regions,  # Number of different patch types (regions)
                image_token_start_index=torch.tensor([0], device=device),
                image_token_end_index=torch.tensor([num_patches], device=device),
                image_token_length=num_patches,
                original_length=num_patches,
                finish_merging=False,
                finish_pruning=False,
                sparsity_list=[]
            )
            
            # Create dummy position embeddings and attention mask for FrameFusion
            position_embeddings = [torch.zeros_like(img_features), torch.zeros_like(img_features)]
            attention_mask = None
            
            # Apply FrameFusion
            reduced_features, _, _ = self.framefusion(
                hidden_states=img_features,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask
            )
            
            # Log reduction statistics
            reduced_patches = reduced_features.shape[1]
            reduction_percentage = (1 - reduced_patches / num_patches) * 100
            print(f'FrameFusion reduced patches from {num_patches} to {reduced_patches} ({reduction_percentage:.2f}% reduction)')
            
            # Apply the adapter to the reduced features
            reduced_features = self.siglip_encoder.adapter(reduced_features)
            results.append(reduced_features)
        
        # If we only had one image, return the result directly
        if len(results) == 1:
            return results[0]
        
        # Otherwise, concatenate the results
        return torch.cat(results, dim=0)

def apply_siglip_framefusion(siglip_encoder, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1, for_single_images=False):
    """
    Apply FrameFusion to a SIGLIP encoder to reduce the number of tokens.
    
    Args:
        siglip_encoder: The SIGLIP encoder to apply FrameFusion to
        cost: The computational budget (higher values allow more token reduction)
        similarity_lower_bound: Threshold for token similarity to be merged
        ratio_lower_bound: Minimum ratio of tokens to keep
        for_single_images: If True, use the adapter optimized for single images
        
    Returns:
        Modified SIGLIP encoder with FrameFusion applied
    """
    # Create a FrameFusion adapter for the SIGLIP encoder
    if for_single_images:
        adapter = SiglipImageFrameFusion(
            siglip_encoder=siglip_encoder,
            cost=cost,
            similarity_lower_bound=similarity_lower_bound,
            ratio_lower_bound=ratio_lower_bound
        )
    else:
        adapter = SiglipFrameFusion(
            siglip_encoder=siglip_encoder,
            cost=cost,
            similarity_lower_bound=similarity_lower_bound,
            ratio_lower_bound=ratio_lower_bound
        )
    
    return adapter
