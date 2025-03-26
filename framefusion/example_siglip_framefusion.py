import os
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from world.encoder.siglip_encoder import SiglipEncoder
from framefusion.siglip_adapter import apply_siglip_framefusion
from framefusion.video_processor import load_video_frames, encode_video_frames_with_framefusion


def parse_args():
    parser = argparse.ArgumentParser(description="Test SIGLIP FrameFusion on a video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--encoder_path", type=str, default="google/siglip2-base-patch16-384", help="Path to the SIGLIP encoder model")
    parser.add_argument("--project_dim", type=int, default=768, help="Projection dimension for the encoder")
    parser.add_argument("--sample_steps", type=int, default=30, help="Number of frames to skip between samples")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract")
    parser.add_argument("--cost", type=float, default=0.3, help="Computational budget for FrameFusion")
    parser.add_argument("--similarity_threshold", type=float, default=0.6, help="Similarity threshold for merging tokens")
    parser.add_argument("--ratio_threshold", type=float, default=0.1, help="Minimum ratio of tokens to keep")
    parser.add_argument("--output_dir", type=str, default="./framefusion_output", help="Directory to save visualization")
    return parser.parse_args()


def visualize_frames(frames, output_path):
    """Visualize a subset of frames and save to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Select a subset of frames if there are too many
    if len(frames) > 16:
        indices = torch.linspace(0, len(frames)-1, 16).long().tolist()
        frames_subset = [frames[i] for i in indices]
    else:
        frames_subset = frames
    
    # Create a grid of images
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, frame in enumerate(frames_subset):
        if i < len(axes):
            axes[i].imshow(frame)
            axes[i].set_title(f"Frame {i}")
            axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(frames_subset), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing video: {args.video_path}")
    
    # Process video without FrameFusion
    print("\n=== Processing without FrameFusion ===")
    result_without_ff = encode_video_frames_with_framefusion(
        args.video_path,
        encoder_path=args.encoder_path,
        project_dim=args.project_dim,
        sample_steps=args.sample_steps,
        max_frames=args.max_frames,
        use_framefusion=False
    )
    encoded_frames_without_ff, frames_without_ff, _ = result_without_ff
    
    # Process video with FrameFusion
    print("\n=== Processing with FrameFusion ===")
    result_with_ff = encode_video_frames_with_framefusion(
        args.video_path,
        encoder_path=args.encoder_path,
        project_dim=args.project_dim,
        sample_steps=args.sample_steps,
        max_frames=args.max_frames,
        use_framefusion=True,
        cost=args.cost,
        similarity_lower_bound=args.similarity_threshold,
        ratio_lower_bound=args.ratio_threshold
    )
    encoded_frames_with_ff, frames_with_ff, original_frame_count = result_with_ff
    
    # Print results
    print("\n=== Results ===")
    print(f"Original frame count: {original_frame_count}")
    print(f"Without FrameFusion: {encoded_frames_without_ff.shape[0]} frames with {encoded_frames_without_ff.shape[1]} patches per frame")
    print(f"With FrameFusion: {encoded_frames_with_ff.shape[0]} frames with {encoded_frames_with_ff.shape[1]} patches per frame")
    print(f"Frame reduction: {(1 - encoded_frames_with_ff.shape[0]/original_frame_count) * 100:.2f}%")
    
    # Visualize frames
    visualize_frames(frames_without_ff, os.path.join(args.output_dir, "original_frames.png"))
    
    print("\nFrameFusion successfully applied to SIGLIP encoder!")


if __name__ == "__main__":
    main()
