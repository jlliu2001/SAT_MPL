"""
Prompt Processing for Language-Based Medical Image Segmentation

This module converts natural language prompts into structured label lists
for the text encoder. Supports both specific label extraction and hierarchical
anatomical group queries.

Example Usage:
    >>> processor = PromptProcessor()

    # Single label query
    >>> processor.parse_prompt("segment amygdala")
    ['amygdala']

    # Multiple labels
    >>> processor.parse_prompt("I want to segment hippocampus and thalamus")
    ['hippocampus', 'thalamus']

    # Hierarchical group query
    >>> processor.parse_prompt("segment whole subcortical part")
    ['hippocampus', 'amygdala', 'caudate', 'putamen', 'pallidum', 'thalamus', 'accumbens']

    # Left/right specification
    >>> processor.parse_prompt("segment left amygdala")
    ['left amygdala']
"""

import re
from typing import List, Dict, Set, Optional, Tuple
import warnings


# Hierarchical brain region groupings
ANATOMICAL_GROUPS = {
    # Subcortical structures (deep brain structures)
    'subcortical': [
        'hippocampus', 'amygdala', 'caudate', 'putamen',
        'pallidum', 'thalamus', 'accumbens'
    ],
    'subcortical_left': [
        'left hippocampus', 'left amygdala', 'left caudate nucleus',
        'left putamen', 'left pallidum', 'left thalamus', 'left nucleus accumbens'
    ],
    'subcortical_right': [
        'right hippocampus', 'right amygdala', 'right caudate nucleus',
        'right putamen', 'right pallidum', 'right thalamus', 'right nucleus accumbens'
    ],

    # Basal ganglia (motor control structures)
    'basal_ganglia': ['caudate', 'putamen', 'pallidum', 'accumbens'],
    'basal_ganglia_left': [
        'left caudate nucleus', 'left putamen', 'left pallidum', 'left nucleus accumbens'
    ],
    'basal_ganglia_right': [
        'right caudate nucleus', 'right putamen', 'right pallidum', 'right nucleus accumbens'
    ],

    # Limbic system (emotion and memory)
    'limbic': ['hippocampus', 'amygdala', 'thalamus'],
    'limbic_left': ['left hippocampus', 'left amygdala', 'left thalamus'],
    'limbic_right': ['right hippocampus', 'right amygdala', 'right thalamus'],

    # Individual structures with full names
    'hippocampus_bilateral': ['left hippocampus', 'right hippocampus'],
    'amygdala_bilateral': ['left amygdala', 'right amygdala'],
    'thalamus_bilateral': ['left thalamus', 'right thalamus'],
    'caudate_bilateral': ['left caudate nucleus', 'right caudate nucleus'],
    'putamen_bilateral': ['left putamen', 'right putamen'],
    'pallidum_bilateral': ['left pallidum', 'right pallidum'],
    'accumbens_bilateral': ['left nucleus accumbens', 'right nucleus accumbens'],
}

# Alias mapping for common terms
GROUP_ALIASES = {
    'subcortical structures': 'subcortical',
    'subcortical regions': 'subcortical',
    'subcortical part': 'subcortical',
    'deep brain structures': 'subcortical',
    'deep structures': 'subcortical',

    'basal ganglia structures': 'basal_ganglia',
    'basal ganglia nuclei': 'basal_ganglia',
    'striatum': 'basal_ganglia',  # Simplified (caudate + putamen + others)

    'limbic system': 'limbic',
    'limbic structures': 'limbic',

    'bilateral hippocampus': 'hippocampus_bilateral',
    'both hippocampi': 'hippocampus_bilateral',
}

# Label name variations and aliases
LABEL_ALIASES = {
    # Basic names (no left/right)
    'hippocampus': 'hippocampus',
    'amygdala': 'amygdala',
    'caudate': 'caudate',
    'caudate nucleus': 'caudate',
    'putamen': 'putamen',
    'pallidum': 'pallidum',
    'globus pallidus': 'pallidum',
    'thalamus': 'thalamus',
    'accumbens': 'accumbens',
    'nucleus accumbens': 'accumbens',

    # Variations with hyphen/underscore
    'caudate-nucleus': 'caudate',
    'caudate_nucleus': 'caudate',
    'globus-pallidus': 'pallidum',
    'globus_pallidus': 'pallidum',
    'nucleus-accumbens': 'accumbens',
    'nucleus_accumbens': 'accumbens',
}


class PromptProcessor:
    """
    Processes natural language prompts for medical image segmentation.

    Converts user queries into standardized label lists compatible with
    the text encoder. Handles various prompt formats including:
    - Direct label mentions: "segment amygdala"
    - Multiple labels: "segment hippocampus and thalamus"
    - Anatomical groups: "segment whole subcortical part"
    - Laterality: "segment left hippocampus"
    """

    def __init__(self,
                 custom_groups: Optional[Dict[str, List[str]]] = None,
                 available_labels: Optional[List[str]] = None):
        """
        Initialize the prompt processor.

        Args:
            custom_groups: Additional anatomical group definitions
            available_labels: List of available labels in your dataset
                            (for validation and suggestions)
        """
        self.anatomical_groups = ANATOMICAL_GROUPS.copy()
        if custom_groups:
            self.anatomical_groups.update(custom_groups)

        self.group_aliases = GROUP_ALIASES.copy()
        self.label_aliases = LABEL_ALIASES.copy()

        # Available labels for validation
        self.available_labels = available_labels or [
            'hippocampus', 'amygdala', 'caudate', 'putamen',
            'pallidum', 'thalamus', 'accumbens'
        ]

    def parse_prompt(self, prompt: str,
                     return_metadata: bool = False) -> List[str]:
        """
        Parse natural language prompt into label list.

        Args:
            prompt: Natural language query
            return_metadata: If True, return (labels, metadata_dict)

        Returns:
            List of standardized label names, or tuple if return_metadata=True

        Examples:
            >>> parse_prompt("segment amygdala")
            ['amygdala']

            >>> parse_prompt("I want to segment left hippocampus and right thalamus")
            ['left hippocampus', 'right thalamus']

            >>> parse_prompt("segment whole subcortical structures")
            ['hippocampus', 'amygdala', 'caudate', 'putamen', 'pallidum', 'thalamus', 'accumbens']
        """
        prompt_lower = prompt.lower().strip()

        metadata = {
            'original_prompt': prompt,
            'laterality': None,  # 'left', 'right', 'bilateral', or None
            'group_detected': None,
            'parsing_method': None,
        }

        # Method 1: Check for anatomical group keywords
        labels, group_name = self._extract_anatomical_group(prompt_lower)
        if labels:
            metadata['group_detected'] = group_name
            metadata['parsing_method'] = 'anatomical_group'
            metadata['laterality'] = self._detect_laterality(prompt_lower)
            return (labels, metadata) if return_metadata else labels

        # Method 2: Extract individual labels with laterality
        labels = self._extract_individual_labels(prompt_lower)
        if labels:
            metadata['parsing_method'] = 'individual_labels'
            # Laterality already encoded in label names
            return (labels, metadata) if return_metadata else labels

        # Method 3: Fallback - try to find any known labels
        labels = self._fallback_label_search(prompt_lower)
        if labels:
            metadata['parsing_method'] = 'fallback_search'
            warnings.warn(f"Using fallback parsing for prompt: '{prompt}'. "
                         f"Consider using more explicit phrasing.")
            return (labels, metadata) if return_metadata else labels

        # No labels found
        raise ValueError(
            f"Could not extract any labels from prompt: '{prompt}'\n"
            f"Available labels: {self.available_labels}\n"
            f"Available groups: {list(self.anatomical_groups.keys())}\n"
            f"Try phrases like: 'segment amygdala', 'segment subcortical structures', "
            f"'segment left hippocampus and right thalamus'"
        )

    def _extract_anatomical_group(self, prompt: str) -> Tuple[List[str], Optional[str]]:
        """
        Extract labels from anatomical group keywords.

        Returns:
            (labels, group_name) tuple, or ([], None) if no group found
        """
        # Check direct group names
        for group_name, labels in self.anatomical_groups.items():
            if group_name.replace('_', ' ') in prompt:
                return labels.copy(), group_name

        # Check aliases
        for alias, group_name in self.group_aliases.items():
            if alias in prompt:
                return self.anatomical_groups[group_name].copy(), group_name

        # Check for "whole" + group patterns
        whole_pattern = r'\b(whole|entire|all)\s+(\w+(?:\s+\w+)*)'
        matches = re.findall(whole_pattern, prompt)
        for _, group_desc in matches:
            # Try to match group description
            group_desc_clean = group_desc.strip()
            if group_desc_clean in self.group_aliases:
                group_name = self.group_aliases[group_desc_clean]
                return self.anatomical_groups[group_name].copy(), group_name
            # Direct match
            group_name = group_desc_clean.replace(' ', '_')
            if group_name in self.anatomical_groups:
                return self.anatomical_groups[group_name].copy(), group_name

        return [], None

    def _extract_individual_labels(self, prompt: str) -> List[str]:
        """
        Extract individual labels with laterality from prompt.

        Handles patterns like:
        - "segment amygdala"
        - "segment left hippocampus and right thalamus"
        - "hippocampus, amygdala, and thalamus"
        """
        extracted_labels = []

        # Split by common delimiters
        # Handle "and", "or", commas, semicolons
        tokens = re.split(r'[,;]|\s+and\s+|\s+or\s+', prompt)

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            # Check for left/right prefix anywhere in token (not just at start)
            laterality = None
            # Look for "left" or "right" followed by a space and label
            left_match = re.search(r'\bleft\s+(\w+(?:\s+\w+)*)', token)
            right_match = re.search(r'\bright\s+(\w+(?:\s+\w+)*)', token)

            if left_match:
                laterality = 'left'
                token_base = left_match.group(1)
            elif right_match:
                laterality = 'right'
                token_base = right_match.group(1)
            else:
                token_base = token

            # Try to match label
            label_found = self._match_label(token_base)
            if label_found:
                # Add laterality if specified
                if laterality:
                    # Expand short names to full form for laterality
                    full_label = self._get_full_label_name(label_found)
                    extracted_labels.append(f"{laterality} {full_label}")
                else:
                    extracted_labels.append(label_found)

        # Deduplicate while preserving order
        return list(dict.fromkeys(extracted_labels))

    def _match_label(self, text: str) -> Optional[str]:
        """
        Match text to a known label.

        Returns standardized label name or None.
        """
        # Remove common prefix words
        text = re.sub(r'^\s*(segment|segmentation of|the|a|an)\s+', '', text)
        text = text.strip()

        # Direct match
        if text in self.available_labels:
            return text

        # Check aliases
        if text in self.label_aliases:
            return self.label_aliases[text]

        # Partial match (word boundary)
        for label in self.available_labels:
            if re.search(r'\b' + re.escape(label) + r'\b', text):
                return label

        return None

    def _fallback_label_search(self, prompt: str) -> List[str]:
        """
        Fallback: search for any occurrence of known labels.
        """
        found_labels = []

        for label in self.available_labels:
            if label in prompt:
                found_labels.append(label)

        return found_labels

    def _detect_laterality(self, prompt: str) -> Optional[str]:
        """Detect if prompt specifies laterality."""
        if 'left' in prompt and 'right' not in prompt:
            return 'left'
        elif 'right' in prompt and 'left' not in prompt:
            return 'right'
        elif ('bilateral' in prompt or
              ('left' in prompt and 'right' in prompt) or
              'both' in prompt):
            return 'bilateral'
        return None

    def _get_full_label_name(self, short_label: str) -> str:
        """
        Convert short label names to full anatomical names.

        E.g., 'caudate' -> 'caudate nucleus'
        """
        full_name_mapping = {
            'caudate': 'caudate nucleus',
            'accumbens': 'nucleus accumbens',
            'pallidum': 'pallidum',
            'hippocampus': 'hippocampus',
            'amygdala': 'amygdala',
            'thalamus': 'thalamus',
            'putamen': 'putamen',
        }
        return full_name_mapping.get(short_label, short_label)

    def get_available_groups(self) -> Dict[str, List[str]]:
        """Get all available anatomical groups."""
        return self.anatomical_groups.copy()

    def get_available_labels(self) -> List[str]:
        """Get all available individual labels."""
        return self.available_labels.copy()

    def suggest_prompts(self) -> List[str]:
        """Generate example prompts for users."""
        examples = [
            # Individual labels
            "segment amygdala",
            "I want to segment hippocampus",
            "segment left thalamus",
            "segment right caudate nucleus",

            # Multiple labels
            "segment hippocampus and amygdala",
            "segment left hippocampus and right thalamus",
            "segment amygdala, hippocampus, and thalamus",

            # Anatomical groups
            "segment whole subcortical structures",
            "segment subcortical part",
            "segment basal ganglia",
            "segment limbic system",
            "segment left subcortical structures",

            # Bilateral
            "segment bilateral hippocampus",
            "segment both amygdalae",
        ]
        return examples


def create_prompt_templates() -> Dict[str, str]:
    """
    Create template prompts for common segmentation tasks.

    Returns:
        Dictionary mapping task names to template strings.
        Use {label} placeholder for dynamic label insertion.
    """
    templates = {
        'single_structure': "Please segment the {label} in this MRI scan.",
        'single_structure_side': "Please segment the {side} {label}.",
        'multiple_structures': "Segment the following structures: {labels}.",
        'group_query': "Segment all {group} structures.",
        'clinical_context': "For surgical planning, segment the {label} and surrounding structures.",
        'comparison': "Segment {label} for volumetric analysis.",
    }
    return templates


# Quick test and examples
if __name__ == "__main__":
    print("=== Prompt Processor Demo ===\n")

    processor = PromptProcessor()

    # Test cases
    test_prompts = [
        "segment amygdala",
        "I want to segment hippocampus",
        "segment left hippocampus and right thalamus",
        "segment whole subcortical structures",
        "segment basal ganglia",
        "segment bilateral hippocampus",
        "hippocampus, amygdala, and thalamus",
    ]

    for prompt in test_prompts:
        try:
            labels, metadata = processor.parse_prompt(prompt, return_metadata=True)
            print(f"Prompt: '{prompt}'")
            print(f"  Labels: {labels}")
            print(f"  Method: {metadata['parsing_method']}")
            print(f"  Laterality: {metadata['laterality']}")
            print()
        except ValueError as e:
            print(f"Error: {e}\n")

    print("\n=== Available Groups ===")
    for group_name, labels in processor.get_available_groups().items():
        print(f"{group_name}: {labels}")

    print("\n=== Example Prompts ===")
    for example in processor.suggest_prompts()[:5]:
        print(f"  - {example}")
