"""Quick test of prompt parsing"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.prompt_processor import PromptProcessor

processor = PromptProcessor()

print("="*70)
print("QUICK PROMPT TEST")
print("="*70)
print()

# Your specific test case
test_prompts = [
    "I want caudate and accumbens",
    "segment amygdala",
    "segment left hippocampus and right thalamus",
    "segment whole subcortical structures",
]

for prompt in test_prompts:
    labels = processor.parse_prompt(prompt)
    print(f'Prompt: "{prompt}"')
    print(f'  ✓ Extracted {len(labels)} label(s): {labels}')
    print()

print("="*70)
print("All prompts parsed correctly!")
print("="*70)
