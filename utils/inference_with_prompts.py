"""
Inference Module with Natural Language Prompt Support

Integrates PromptProcessor with the Text_Encoder for interactive segmentation.
Allows users to specify segmentation targets using natural language prompts
instead of explicit label lists.

Example Usage:
    >>> from utils.inference_with_prompts import PromptBasedSegmentation
    >>> from model.text_encoder import Text_Encoder

    >>> # Initialize
    >>> text_encoder = Text_Encoder(text_encoder='ours', checkpoint='path/to/checkpoint')
    >>> prompt_seg = PromptBasedSegmentation(text_encoder)

    >>> # Use natural language
    >>> queries = prompt_seg.encode_prompt(
    ...     "segment left hippocampus and right amygdala",
    ...     modality="mri"
    ... )
    >>> # queries is now ready for your decoder

    >>> # Use with image
    >>> result = prompt_seg.segment_from_prompt(
    ...     prompt="segment whole subcortical structures",
    ...     image=brain_mri,
    ...     model=your_segmentation_model,
    ...     modality="mri"
    ... )
"""

import torch
from typing import List, Dict, Optional, Union, Tuple
import warnings

from .prompt_processor import PromptProcessor
from utils.brain_region_mapping import standardize_brain_labels


class PromptBasedSegmentation:
    """
    Wrapper for text encoder with natural language prompt support.

    Converts user prompts to label lists, standardizes labels,
    and generates text embeddings for the segmentation model.
    """

    def __init__(self,
                 text_encoder,
                 prompt_processor: Optional[PromptProcessor] = None,
                 standardize_labels: bool = True,
                 device: Optional[torch.device] = None):
        """
        Initialize prompt-based segmentation.

        Args:
            text_encoder: Initialized Text_Encoder instance
            prompt_processor: Custom PromptProcessor (uses default if None)
            standardize_labels: Whether to standardize labels to SAT format
            device: Device for text encoder
        """
        self.text_encoder = text_encoder
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize prompt processor
        if prompt_processor is None:
            # Default labels from your dataset
            default_labels = [
                'hippocampus', 'amygdala', 'caudate', 'putamen',
                'pallidum', 'thalamus', 'accumbens'
            ]
            prompt_processor = PromptProcessor(available_labels=default_labels)

        self.prompt_processor = prompt_processor
        self.standardize_labels = standardize_labels

    def encode_prompt(self,
                     prompt: str,
                     modality: str,
                     return_label_list: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """
        Convert natural language prompt to text embeddings.

        Args:
            prompt: Natural language query (e.g., "segment amygdala")
            modality: Imaging modality ('mri', 'ct', 'pet', 'us')
            return_label_list: If True, also return the extracted label list

        Returns:
            Text embedding queries tensor [N, D], or (queries, labels) if return_label_list=True

        Example:
            >>> queries = encoder.encode_prompt("segment left hippocampus", "mri")
            >>> queries.shape  # torch.Size([1, 768]) - one query for one label
        """
        # Parse prompt to get labels
        try:
            labels = self.prompt_processor.parse_prompt(prompt)
        except ValueError as e:
            raise ValueError(f"Prompt parsing failed: {e}")

        # Standardize labels if needed
        if self.standardize_labels:
            labels = standardize_brain_labels(labels)

        # Convert modality to lowercase
        modality = modality.lower()

        # Encode with text encoder
        # text_encoder.forward expects (label_name, modality_name)
        # where label_name is List[str] and modality_name is str
        queries = self.text_encoder(labels, modality)  # [N, D]

        if return_label_list:
            return queries, labels
        return queries

    def encode_multiple_prompts(self,
                                prompts: List[str],
                                modality: str,
                                return_label_lists: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[List[str]]]]:
        """
        Encode multiple prompts in batch.

        Args:
            prompts: List of natural language queries
            modality: Imaging modality
            return_label_lists: If True, return label lists for each prompt

        Returns:
            Batched queries [B, N, D] or (queries, label_lists) if return_label_lists=True
        """
        all_queries = []
        all_labels = []

        for prompt in prompts:
            queries, labels = self.encode_prompt(prompt, modality, return_label_list=True)
            all_queries.append(queries)
            all_labels.append(labels)

        # Pad to same length if necessary (in case different prompts have different N)
        max_n = max(q.shape[0] for q in all_queries)

        padded_queries = []
        for queries in all_queries:
            n, d = queries.shape
            if n < max_n:
                # Pad with zeros
                padding = torch.zeros(max_n - n, d, device=queries.device)
                queries = torch.cat([queries, padding], dim=0)
            padded_queries.append(queries)

        batched_queries = torch.stack(padded_queries)  # [B, N, D]

        if return_label_lists:
            return batched_queries, all_labels
        return batched_queries

    def segment_from_prompt(self,
                           prompt: str,
                           image: torch.Tensor,
                           model,
                           modality: str = "mri",
                           return_labels: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """
        Complete segmentation pipeline from prompt.

        Args:
            prompt: Natural language segmentation query
            image: Input image tensor [C, H, W, D] or [B, C, H, W, D]
            model: Segmentation model with forward(image, text_queries) interface
            modality: Imaging modality
            return_labels: If True, return (mask, labels)

        Returns:
            Segmentation mask tensor, or (mask, labels) if return_labels=True

        Example:
            >>> mask = seg.segment_from_prompt(
            ...     "segment subcortical structures",
            ...     brain_image,
            ...     segmentation_model,
            ...     modality="mri"
            ... )
        """
        # Get text queries from prompt
        queries, labels = self.encode_prompt(prompt, modality, return_label_list=True)

        # Run segmentation model
        with torch.no_grad():
            # Assuming model.forward(image, text_queries) -> mask
            mask = model(image, queries)

        if return_labels:
            return mask, labels
        return mask

    def interactive_prompt(self, modality: str = "mri"):
        """
        Interactive prompt interface for testing.

        Usage:
            >>> seg.interactive_prompt()
            Enter prompt (or 'quit' to exit): segment left hippocampus
            Parsed labels: ['left hippocampus']
            Queries shape: torch.Size([1, 768])
        """
        print("=== Interactive Prompt Interface ===")
        print("Enter segmentation prompts in natural language.")
        print("Type 'examples' to see example prompts.")
        print("Type 'groups' to see available anatomical groups.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                prompt = input("Enter prompt: ").strip()

                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'examples':
                    print("\nExample prompts:")
                    for example in self.prompt_processor.suggest_prompts():
                        print(f"  - {example}")
                    print()
                    continue
                elif prompt.lower() == 'groups':
                    print("\nAvailable anatomical groups:")
                    for group, labels in self.prompt_processor.get_available_groups().items():
                        print(f"  {group}: {', '.join(labels[:3])}{'...' if len(labels) > 3 else ''}")
                    print()
                    continue
                elif not prompt:
                    continue

                # Process prompt
                queries, labels = self.encode_prompt(prompt, modality, return_label_list=True)

                print(f"✓ Parsed labels: {labels}")
                print(f"✓ Queries shape: {queries.shape}")
                print(f"✓ Ready for segmentation with {len(labels)} structure(s)\n")

            except ValueError as e:
                print(f"✗ Error: {e}\n")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    def validate_prompt(self, prompt: str) -> Dict:
        """
        Validate and analyze a prompt without encoding.

        Args:
            prompt: Natural language query

        Returns:
            Dictionary with parsing information and validation results
        """
        try:
            labels, metadata = self.prompt_processor.parse_prompt(prompt, return_metadata=True)

            # Check if labels are in standard format
            if self.standardize_labels:
                standardized = standardize_brain_labels(labels)
            else:
                standardized = labels

            return {
                'valid': True,
                'original_prompt': prompt,
                'extracted_labels': labels,
                'standardized_labels': standardized,
                'num_structures': len(labels),
                'parsing_method': metadata['parsing_method'],
                'laterality': metadata['laterality'],
                'group_detected': metadata['group_detected'],
            }

        except ValueError as e:
            return {
                'valid': False,
                'error': str(e),
                'original_prompt': prompt,
            }


def create_inference_pipeline(text_encoder_checkpoint: str,
                              text_encoder_type: str = 'ours',
                              device: Optional[torch.device] = None) -> PromptBasedSegmentation:
    """
    Factory function to create a complete prompt-based segmentation pipeline.

    Args:
        text_encoder_checkpoint: Path to text encoder checkpoint
        text_encoder_type: Type of text encoder ('ours', 'medcpt', 'basebert')
        device: Device to use

    Returns:
        Initialized PromptBasedSegmentation instance

    Example:
        >>> pipeline = create_inference_pipeline(
        ...     text_encoder_checkpoint='path/to/checkpoint.pth',
        ...     text_encoder_type='ours'
        ... )
        >>> mask = pipeline.segment_from_prompt(
        ...     "segment hippocampus",
        ...     brain_image,
        ...     model
        ... )
    """
    from model.text_encoder import Text_Encoder

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize text encoder
    # Note: Text_Encoder expects gpu_id for DDP, so we extract device index
    gpu_id = device.index if device.type == 'cuda' else 0

    text_encoder = Text_Encoder(
        text_encoder=text_encoder_type,
        checkpoint=text_encoder_checkpoint,
        device=device,
        gpu_id=gpu_id
    )

    # Create prompt-based segmentation wrapper
    prompt_seg = PromptBasedSegmentation(
        text_encoder=text_encoder,
        device=device
    )

    return prompt_seg


# Example usage and testing
if __name__ == "__main__":
    print("=== Prompt-Based Segmentation Demo ===\n")

    # For demonstration without actual model
    class MockTextEncoder:
        """Mock text encoder for testing"""
        def __call__(self, labels, modality):
            # Return dummy embeddings
            return torch.randn(len(labels), 768)

    # Create instance with mock encoder
    mock_encoder = MockTextEncoder()
    prompt_seg = PromptBasedSegmentation(mock_encoder)

    # Test various prompts
    test_cases = [
        ("segment amygdala", "mri"),
        ("segment left hippocampus and right thalamus", "mri"),
        ("segment whole subcortical structures", "mri"),
        ("I want to segment basal ganglia", "mri"),
    ]

    for prompt, modality in test_cases:
        print(f"Prompt: '{prompt}'")
        validation = prompt_seg.validate_prompt(prompt)
        if validation['valid']:
            print(f"  ✓ Valid")
            print(f"    Labels: {validation['extracted_labels']}")
            print(f"    Method: {validation['parsing_method']}")
            print(f"    Structures: {validation['num_structures']}")

            # Try encoding
            queries = prompt_seg.encode_prompt(prompt, modality)
            print(f"    Queries shape: {queries.shape}")
        else:
            print(f"  ✗ Invalid: {validation['error']}")
        print()

    print("\n=== Interactive Mode ===")
    print("Uncomment the following line to try interactive mode:")
    print("# prompt_seg.interactive_prompt()")
