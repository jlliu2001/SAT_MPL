"""
Example: Prompt-Based Inference for Brain MRI Segmentation

This script demonstrates how to use natural language prompts with your
trained SAT-MPL model for interactive segmentation.

Usage:
    # Single prompt inference
    python examples/prompt_based_inference_example.py \
        --checkpoint path/to/model.pth \
        --text_encoder_checkpoint path/to/text_encoder.pth \
        --image_path path/to/mri.nii.gz \
        --prompt "segment left hippocampus and amygdala"

    # Interactive mode
    python examples/prompt_based_inference_example.py \
        --checkpoint path/to/model.pth \
        --text_encoder_checkpoint path/to/text_encoder.pth \
        --interactive
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import nibabel as nib
from pathlib import Path

from utils.inference_with_prompts import PromptBasedSegmentation, create_inference_pipeline
from utils.prompt_processor import PromptProcessor
from model.text_encoder import Text_Encoder


def load_mri_image(image_path: str) -> torch.Tensor:
    """
    Load MRI image from file.

    Args:
        image_path: Path to NIfTI file (.nii or .nii.gz)

    Returns:
        Image tensor [1, H, W, D] (with channel dimension)
    """
    # Load using nibabel
    img = nib.load(image_path)
    data = img.get_fdata()

    # Convert to tensor and add channel dimension
    tensor = torch.from_numpy(data).float().unsqueeze(0)  # [1, H, W, D]

    return tensor


def save_segmentation_mask(mask: torch.Tensor,
                           output_path: str,
                           reference_image_path: str = None):
    """
    Save segmentation mask to NIfTI file.

    Args:
        mask: Segmentation mask tensor [N, H, W, D]
        output_path: Output file path
        reference_image_path: Reference image for header info
    """
    # Convert to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # If multiple labels, combine into single volume with different values
    if mask.ndim == 4 and mask.shape[0] > 1:
        # Each channel becomes a different label value
        combined_mask = np.zeros(mask.shape[1:], dtype=np.int16)
        for i in range(mask.shape[0]):
            combined_mask[mask[i] > 0.5] = i + 1
        mask = combined_mask
    elif mask.ndim == 4:
        mask = mask[0]  # Remove channel dimension if single label

    # Create NIfTI image
    if reference_image_path:
        # Use reference header
        ref_img = nib.load(reference_image_path)
        seg_img = nib.Nifti1Image(mask, ref_img.affine, ref_img.header)
    else:
        # Create new header
        seg_img = nib.Nifti1Image(mask, np.eye(4))

    # Save
    nib.save(seg_img, output_path)
    print(f"Segmentation saved to: {output_path}")


def single_prompt_inference(args):
    """Run inference with a single prompt."""
    print("=== Single Prompt Inference ===\n")

    # Initialize pipeline
    print("Loading models...")
    # Note: This is a simplified version. You'll need to load your full model
    # For demonstration, we show the text encoder part
    from model.text_encoder import Text_Encoder

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        device=device,
        gpu_id=args.gpu
    )

    prompt_seg = PromptBasedSegmentation(text_encoder, device=device)

    # Validate prompt
    print(f"\nPrompt: '{args.prompt}'")
    validation = prompt_seg.validate_prompt(args.prompt)

    if not validation['valid']:
        print(f"Error: {validation['error']}")
        return

    print(f"✓ Valid prompt")
    print(f"  Extracted labels: {validation['extracted_labels']}")
    print(f"  Standardized: {validation['standardized_labels']}")
    print(f"  Parsing method: {validation['parsing_method']}")
    print(f"  Number of structures: {validation['num_structures']}")

    # Load image
    print(f"\nLoading image from: {args.image_path}")
    image = load_mri_image(args.image_path)
    print(f"Image shape: {image.shape}")

    # Generate text queries
    print("\nGenerating text embeddings...")
    queries, labels = prompt_seg.encode_prompt(
        args.prompt,
        args.modality,
        return_label_list=True
    )
    print(f"Text queries shape: {queries.shape}")
    print(f"Labels for segmentation: {labels}")

    print("\n✓ Text queries ready for segmentation model")
    print("  Note: Full segmentation requires loading the complete SAT-MPL model")
    print("  See evaluate_mplseg.py for complete inference pipeline")

    # TODO: Load full model and run inference
    # mask = model(image, queries)
    # save_segmentation_mask(mask, args.output, args.image_path)


def interactive_mode(args):
    """Run interactive prompt interface."""
    print("=== Interactive Prompt Mode ===\n")

    # Initialize
    print("Loading text encoder...")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        device=device,
        gpu_id=args.gpu
    )

    prompt_seg = PromptBasedSegmentation(text_encoder, device=device)

    print("Ready!\n")

    # Run interactive interface
    prompt_seg.interactive_prompt(modality=args.modality)


def demo_mode():
    """Demonstrate prompt processing without loading models."""
    print("=== Prompt Processing Demo (No Model Loading) ===\n")

    processor = PromptProcessor()

    print("Available anatomical groups:")
    for group, labels in list(processor.get_available_groups().items())[:5]:
        print(f"  - {group}: {labels}")
    print("  ... (more groups available)\n")

    print("Example prompts:")
    test_prompts = [
        "segment amygdala",
        "segment left hippocampus and right thalamus",
        "segment whole subcortical structures",
        "I want to segment basal ganglia",
        "segment bilateral hippocampus",
    ]

    for prompt in test_prompts:
        try:
            labels, metadata = processor.parse_prompt(prompt, return_metadata=True)
            print(f"\nPrompt: '{prompt}'")
            print(f"  → Labels: {labels}")
            print(f"  → Method: {metadata['parsing_method']}")
            print(f"  → Laterality: {metadata.get('laterality', 'N/A')}")
        except ValueError as e:
            print(f"\nPrompt: '{prompt}'")
            print(f"  → Error: {e}")

    print("\n" + "="*60)
    print("To use with your model, run:")
    print("  python examples/prompt_based_inference_example.py \\")
    print("    --checkpoint path/to/model.pth \\")
    print("    --text_encoder_checkpoint path/to/text_encoder.pth \\")
    print("    --interactive")


def main():
    parser = argparse.ArgumentParser(
        description="Prompt-based inference for brain MRI segmentation"
    )

    # Model arguments
    parser.add_argument('--checkpoint', type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('--text_encoder_checkpoint', type=str,
                       help='Path to text encoder checkpoint')
    parser.add_argument('--text_encoder', type=str, default='ours',
                       choices=['ours', 'medcpt', 'basebert'],
                       help='Text encoder type')

    # Inference arguments
    parser.add_argument('--image_path', type=str,
                       help='Path to input MRI image (.nii.gz)')
    parser.add_argument('--prompt', type=str,
                       help='Natural language segmentation prompt')
    parser.add_argument('--modality', type=str, default='mri',
                       choices=['mri', 'ct', 'pet', 'us'],
                       help='Imaging modality')
    parser.add_argument('--output', type=str,
                       help='Output path for segmentation mask')

    # Mode selection
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo without loading models')

    # Device
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')

    args = parser.parse_args()

    # Determine mode
    if args.demo:
        demo_mode()
    elif args.interactive:
        if not args.text_encoder_checkpoint:
            print("Error: --text_encoder_checkpoint required for interactive mode")
            sys.exit(1)
        interactive_mode(args)
    elif args.prompt and args.image_path:
        if not args.text_encoder_checkpoint:
            print("Error: --text_encoder_checkpoint required for inference")
            sys.exit(1)
        single_prompt_inference(args)
    else:
        print("Error: Must specify either --demo, --interactive, or both --prompt and --image_path")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
