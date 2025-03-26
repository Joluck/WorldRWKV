import argparse
from infer.worldmodel import Worldinfer
from PIL import Image
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Generate text from an image using RWKV with optional FrameFusion')
    parser.add_argument('--image_path', type=str, default='./docs/03-Confusing-Pictures.jpg', help='Path to the input image')
    parser.add_argument('--llm_path', type=str, default='/home/rwkv/models/RWKV7-3B-siglip2/rwkv-0', help='Path to RWKV model')
    parser.add_argument('--encoder_path', type=str, default='google/siglip2-base-patch16-384', help='Path to encoder model')
    parser.add_argument('--use_framefusion', action='store_true', help='Use FrameFusion to reduce tokens')
    parser.add_argument('--cost', type=float, default=0.3, help='FrameFusion cost parameter')
    parser.add_argument('--similarity_threshold', type=float, default=0.6, help='FrameFusion similarity threshold')
    parser.add_argument('--ratio_threshold', type=float, default=0.1, help='FrameFusion ratio threshold')
    parser.add_argument('--prompt', type=str, default='\x16User: What is unusual about this image?\x17Assistant:', help='Prompt for the model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the image
    img_path = args.image_path
    image = Image.open(img_path).convert('RGB')
    
    # Prepare token reduction parameters if needed
    if args.use_framefusion:
        print(f"Using FrameFusion with cost={args.cost}, similarity_threshold={args.similarity_threshold}, ratio_threshold={args.ratio_threshold}")
        
        # Configure token reduction parameters
        token_reduction_params = {
            'cost': args.cost,
            'similarity_threshold': args.similarity_threshold,
            'ratio_threshold': args.ratio_threshold,
            'for_single_images': True  # Apply to single images
        }
        
        # Create model with built-in token reduction
        model = Worldinfer(
            model_path=args.llm_path, 
            encoder_type='siglip', 
            encoder_path=args.encoder_path,
            use_token_reduction=True,
            token_reduction_params=token_reduction_params
        )
    else:
        print("Using standard SIGLIP encoder without FrameFusion")
        model = Worldinfer(model_path=args.llm_path, encoder_type='siglip', encoder_path=args.encoder_path)
    
    # Generate text from the image
    result, _ = model.generate(args.prompt, image)
    
    print("\nGenerated Response:\n")
    print(result)

# Function removed as we now use built-in token reduction in Worldinfer

if __name__ == '__main__':
    main()
