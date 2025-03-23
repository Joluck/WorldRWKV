import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from world.encoder.siglip_encoder import SiglipEncoder
from framefusion.siglip_adapter import apply_siglip_framefusion
from infer.worldmodel import Worldinfer


def load_video_frames(video_path, sample_steps=None, max_frames=None, keyframe_timestamps=None):
    """
    Load frames from a video based on sample steps or timestamps
    
    Args:
        video_path: Path to the video file
        sample_steps: Number of frames to skip between samples (e.g., sample_steps=30 means take 1 frame per second in a 30fps video)
        max_frames: Maximum number of frames to extract (None means no limit)
        keyframe_timestamps: Optional list of timestamps (in seconds) to extract specific frames from
        
    Returns:
        List of PIL Images corresponding to the sampled frames
    """
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties: {fps} fps, {total_frames} frames, {duration:.2f} seconds")
    
    frames = []
    
    # Case 1: Extract frames using sample_steps
    if sample_steps is not None:
        # Calculate frame indices to sample
        frame_indices = list(range(0, total_frames, sample_steps))
        
        # Limit to max_frames if specified
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
            
        print(f"Sampling {len(frame_indices)} frames with step size {sample_steps}")
        
        for frame_idx in frame_indices:
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame at index {frame_idx}")
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    # Case 2: Extract frames at specified timestamps
    elif keyframe_timestamps is not None:
        print(f"Extracting {len(keyframe_timestamps)} frames at specified timestamps")
        
        for timestamp in keyframe_timestamps:
            # Convert timestamp to frame number
            frame_number = int(timestamp * fps)
            
            # Check if frame number is valid
            if frame_number >= total_frames:
                print(f"Warning: Timestamp {timestamp}s exceeds video duration {duration:.2f}s")
                continue
            
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame at timestamp {timestamp}s")
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    # Release the video capture
    cap.release()
    
    return frames


def encode_video_frames_with_framefusion(video_path, caption_json_path=None, 
                                       encoder_path='google/siglip2-base-patch16-384', 
                                       project_dim=768, sample_steps=None, max_frames=None,
                                       use_framefusion=True, cost=0.3, 
                                       similarity_lower_bound=0.6, ratio_lower_bound=0.1):
    """
    Process a video by extracting frames and encoding them with SiglipEncoder,
    optionally applying FrameFusion to reduce the number of tokens
    
    Args:
        video_path: Path to the video file
        caption_json_path: Path to the JSON file containing keyframe timestamps (optional)
        encoder_path: Path to the SiglipEncoder model
        project_dim: Projection dimension for the encoder
        sample_steps: Number of frames to skip between samples
        max_frames: Maximum number of frames to extract (None means no limit)
        use_framefusion: Whether to apply FrameFusion to reduce tokens
        cost: The computational budget for FrameFusion
        similarity_lower_bound: Threshold for token similarity to be merged
        ratio_lower_bound: Minimum ratio of tokens to keep
        
    Returns:
        Tuple of (encoded_frames, frames, original_frame_count)
    """
    # Determine how to extract frames
    keyframe_timestamps = None
    
    if caption_json_path is not None:
        # Load keyframe timestamps from JSON
        with open(caption_json_path, 'r') as f:
            caption_data = json.load(f)
        
        keyframe_timestamps = caption_data.get('keyframe', [])
        if not keyframe_timestamps:
            print(f"No keyframe timestamps found in {caption_json_path}, using sample_steps instead")
            keyframe_timestamps = None
        else:
            print(f"Found {len(keyframe_timestamps)} keyframe timestamps")
    
    # Load frames from video
    frames = load_video_frames(video_path, sample_steps=sample_steps, max_frames=max_frames, 
                             keyframe_timestamps=keyframe_timestamps)
    print(f"Extracted {len(frames)} frames from video")
    original_frame_count = len(frames)
    
    # Initialize the SiglipEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    encoder = SiglipEncoder(encoder_path, project_dim, device=device)
    
    # Apply FrameFusion if requested
    if use_framefusion and len(frames) > 1:
        print(f"Applying FrameFusion with cost={cost}, similarity_threshold={similarity_lower_bound}, ratio_threshold={ratio_lower_bound}")
        framefusion_encoder = apply_siglip_framefusion(
            encoder, 
            cost=cost, 
            similarity_lower_bound=similarity_lower_bound, 
            ratio_lower_bound=ratio_lower_bound
        )
        
        # Encode the frames with FrameFusion
        with torch.no_grad():
            encoded_frames = framefusion_encoder.forward_with_framefusion(frames)
        
        print(f"Original frames: {original_frame_count}, After FrameFusion: {encoded_frames.shape}")
        print(f"Frame reduction: {(1 - encoded_frames.shape[0]/original_frame_count) * 100:.2f}%")
    else:
        # Encode the frames without FrameFusion
        with torch.no_grad():
            encoded_frames = encoder(frames)
        
        print(f"Encoded frames shape: {encoded_frames.shape}")
    
    return encoded_frames, frames, original_frame_count


def process_video_with_framefusion(video_path, caption_json_path=None, 
                                 encoder_path='google/siglip2-base-patch16-384', 
                                 project_dim=768, sample_steps=None, max_frames=None,
                                 use_framefusion=True, cost=0.3, 
                                 similarity_lower_bound=0.6, ratio_lower_bound=0.1):
    """
    Process a video by extracting frames and encoding them with SiglipEncoder,
    optionally applying FrameFusion to reduce the number of tokens
    
    Args:
        video_path: Path to the video file
        caption_json_path: Path to the JSON file containing keyframe timestamps (optional)
        encoder_path: Path to the SiglipEncoder model
        project_dim: Projection dimension for the encoder
        sample_steps: Number of frames to skip between samples
        max_frames: Maximum number of frames to extract (None means no limit)
        use_framefusion: Whether to apply FrameFusion to reduce tokens
        cost: The computational budget for FrameFusion
        similarity_lower_bound: Threshold for token similarity to be merged
        ratio_lower_bound: Minimum ratio of tokens to keep
        
    Returns:
        Dictionary containing encoded frames, original frames, and metadata
    """
    # Encode video frames with optional FrameFusion
    encoded_frames, frames, original_frame_count = encode_video_frames_with_framefusion(
        video_path, 
        caption_json_path=caption_json_path,
        encoder_path=encoder_path, 
        project_dim=project_dim, 
        sample_steps=sample_steps, 
        max_frames=max_frames,
        use_framefusion=use_framefusion, 
        cost=cost, 
        similarity_lower_bound=similarity_lower_bound, 
        ratio_lower_bound=ratio_lower_bound
    )
    
    # Return results
    return {
        'encoded_frames': encoded_frames,
        'frames': frames,
        'original_frame_count': original_frame_count,
        'final_frame_count': encoded_frames.shape[1],
        'reduction_percentage': (1 - encoded_frames.shape[1]/original_frame_count) * 100 if original_frame_count > 1 else 0
    }
