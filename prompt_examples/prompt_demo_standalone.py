"""
Standalone Prompt Processing Demo

This script demonstrates prompt processing without requiring PyTorch or any models.
Perfect for testing and understanding the prompt system before integrating with your model.

Usage:
    python examples/prompt_demo_standalone.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.prompt_processor import PromptProcessor


def demo_basic_prompts():
    """Demonstrate basic prompt parsing"""
    print("="*70)
    print("BASIC PROMPT PARSING EXAMPLES")
    print("="*70)
    print()

    processor = PromptProcessor()

    test_prompts = [
        # Single structures
        "segment amygdala",
        "I want to segment hippocampus",
        "segment the thalamus",

        # Multiple structures
        "segment hippocampus and amygdala",
        "segment hippocampus, amygdala, and thalamus",

        # With laterality
        "segment left hippocampus",
        "segment right amygdala",
        "segment left hippocampus and right thalamus",

        # Anatomical groups
        "segment whole subcortical structures",
        "segment basal ganglia",
        "segment limbic system",

        # More complex
        "I want to segment bilateral hippocampus",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        try:
            labels, metadata = processor.parse_prompt(prompt, return_metadata=True)
            print(f"{i}. Prompt: \"{prompt}\"")
            print(f"   → Labels: {labels}")
            print(f"   → Method: {metadata['parsing_method']}")
            if metadata['group_detected']:
                print(f"   → Group: {metadata['group_detected']}")
            print()
        except ValueError as e:
            print(f"{i}. Prompt: \"{prompt}\"")
            print(f"   → ERROR: {str(e)[:80]}...")
            print()


def demo_anatomical_groups():
    """Show all available anatomical groups"""
    print("="*70)
    print("AVAILABLE ANATOMICAL GROUPS")
    print("="*70)
    print()

    processor = PromptProcessor()
    groups = processor.get_available_groups()

    # Show main groups
    main_groups = ['subcortical', 'basal_ganglia', 'limbic']

    for group_name in main_groups:
        if group_name in groups:
            labels = groups[group_name]
            print(f"• {group_name.upper().replace('_', ' ')}")
            print(f"  Structures: {', '.join(labels)}")
            print(f"  Example prompt: \"segment {group_name.replace('_', ' ')}\"")
            print()

    print(f"Total available groups: {len(groups)}")
    print()


def demo_interactive():
    """Interactive prompt testing"""
    print("="*70)
    print("INTERACTIVE PROMPT TESTING")
    print("="*70)
    print()
    print("Enter prompts to see how they are parsed.")
    print("Commands:")
    print("  'examples' - Show example prompts")
    print("  'groups'   - Show available groups")
    print("  'labels'   - Show available labels")
    print("  'quit'     - Exit")
    print()

    processor = PromptProcessor()

    while True:
        try:
            prompt = input("Enter prompt: ").strip()

            if not prompt:
                continue
            elif prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'examples':
                print("\nExample prompts:")
                for ex in processor.suggest_prompts():
                    print(f"  • {ex}")
                print()
                continue
            elif prompt.lower() == 'groups':
                print("\nAvailable groups:")
                groups = processor.get_available_groups()
                for group in ['subcortical', 'basal_ganglia', 'limbic']:
                    if group in groups:
                        print(f"  • {group}: {', '.join(groups[group][:3])}...")
                print()
                continue
            elif prompt.lower() == 'labels':
                print("\nAvailable labels:")
                for label in processor.get_available_labels():
                    print(f"  • {label}")
                print()
                continue

            # Parse prompt
            labels, metadata = processor.parse_prompt(prompt, return_metadata=True)

            print(f"✓ Success!")
            print(f"  Extracted labels: {labels}")
            print(f"  Number of structures: {len(labels)}")
            print(f"  Parsing method: {metadata['parsing_method']}")
            if metadata['group_detected']:
                print(f"  Anatomical group: {metadata['group_detected']}")
            print()

        except ValueError as e:
            print(f"✗ Error: {str(e)[:100]}")
            print()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


def demo_use_case_examples():
    """Show practical use case examples"""
    print("="*70)
    print("PRACTICAL USE CASE EXAMPLES")
    print("="*70)
    print()

    processor = PromptProcessor()

    use_cases = [
        {
            'scenario': 'Surgical Planning - Temporal Lobe Epilepsy',
            'prompt': 'segment left hippocampus and left amygdala',
            'rationale': 'Need to visualize structures for epilepsy surgery planning'
        },
        {
            'scenario': "Parkinson's Disease - Motor Circuit Assessment",
            'prompt': 'segment basal ganglia',
            'rationale': 'Evaluate motor control nuclei affected by Parkinson\'s'
        },
        {
            'scenario': 'Memory Research - Volumetric Analysis',
            'prompt': 'segment bilateral hippocampus',
            'rationale': 'Measure hippocampal volume in both hemispheres'
        },
        {
            'scenario': 'Comprehensive Deep Brain Analysis',
            'prompt': 'segment whole subcortical structures',
            'rationale': 'Complete segmentation of all subcortical regions'
        },
    ]

    for i, use_case in enumerate(use_cases, 1):
        print(f"{i}. {use_case['scenario']}")
        print(f"   Rationale: {use_case['rationale']}")
        print(f"   Prompt: \"{use_case['prompt']}\"")

        labels = processor.parse_prompt(use_case['prompt'])
        print(f"   → Will segment: {', '.join(labels)}")
        print()


def main():
    """Main demo function"""
    import sys

    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  PROMPT-BASED BRAIN SEGMENTATION - DEMO                          ║")
    print("║  Natural Language Interface for SAT-MPL                           ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        demo_interactive()
    else:
        # Run all demos
        demo_basic_prompts()
        input("Press Enter to continue...")
        print()

        demo_anatomical_groups()

        input("Press Enter to continue...")
        print()

        demo_use_case_examples()
        input("Press Enter to continue...")
        print()

        # Offer interactive mode
        print("="*70)
        print("Would you like to try interactive mode? (y/n)")
        response = input("> ").strip().lower()
        if response in ['y', 'yes']:
            print()
            demo_interactive()
        else:
            print("\nTo run interactive mode later, use:")
            print("  python examples/prompt_demo_standalone.py --interactive")

    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Review the documentation: docs/PROMPT_USAGE.md")
    print("  2. Integrate with your model using utils/inference_with_prompts.py")
    print("  3. Test with your MRI data")
    print()


if __name__ == "__main__":
    main()
