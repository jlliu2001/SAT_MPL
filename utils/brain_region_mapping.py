"""
Brain Region Label Standardization for SAT Framework
Maps private brain segmentation labels to SAT standard terminology (Brain_Atlas format)
"""

import warnings
from typing import List, Dict, Optional

# Standard brain region labels from SAT Brain_Atlas dataset (no modality prefix)
SAT_BRAIN_LABELS = {
    "left hippocampus", "right hippocampus", "hippocampus",
    "left amygdala", "right amygdala", "amygdala",
    "left anterior temporal lobe medial part", "right anterior temporal lobe medial part",
    "left anterior temporal lobe lateral part", "right anterior temporal lobe lateral part",
    "left parahippocampal and ambient gyrus", "right parahippocampal and ambient gyrus",
    "left superior temporal gyrus middle part", "right superior temporal gyrus middle part",
    "left middle and inferior temporal gyrus", "right middle and inferior temporal gyrus",
    "left fusiform gyrus", "right fusiform gyrus",
    "left cerebellum", "right cerebellum", "cerebellum",
    "brainstem excluding substantia nigra", "brainstem",
    "right insula posterior long gyrus", "left insula posterior long gyrus",
    "right lateral remainder occipital lobe", "left lateral remainder occipital lobe",
    "right anterior cingulate gyrus", "left anterior cingulate gyrus",
    "right posterior cingulate gyrus", "left posterior cingulate gyrus",
    "right middle frontal gyrus", "left middle frontal gyrus",
    "right posterior temporal lobe", "left posterior temporal lobe",
    "right angular gyrus", "left angular gyrus",
    "right caudate nucleus", "left caudate nucleus",
    "right nucleus accumbens", "left nucleus accumbens",
    "right putamen", "left putamen",
    "right thalamus", "left thalamus", "thalamus",
    "right pallidum", "left pallidum",
    "corpus callosum",
    "left Lateral ventricle excluding temporal horn", "right Lateral ventricle excluding temporal horn",
    "left Lateral ventricle temporal horn", "right Lateral ventricle temporal horn",
    "Third ventricle", "lateral ventricle",
    "right precentral gyrus", "left precentral gyrus",
    "right straight gyrus", "left straight gyrus",
    "right anterior orbital gyrus", "left anterior orbital gyrus",
    "right inferior frontal gyrus", "left inferior frontal gyrus",
    "right superior frontal gyrus", "left superior frontal gyrus",
    "right postcentral gyrus", "left postcentral gyrus",
    "right superior parietal gyrus", "left superior parietal gyrus",
    "right lingual gyrus", "left lingual gyrus",
    "right cuneus", "left cuneus",
    "right medial orbital gyrus", "left medial orbital gyrus",
    "right lateral orbital gyrus", "left lateral orbital gyrus",
    "right posterior orbital gyrus", "left posterior orbital gyrus",
    "right substantia nigra", "left substantia nigra",
    "right subgenual frontal cortex", "left subgenual frontal cortex",
    "right subcallosal area", "left subcallosal area",
    "right pre-subgenual frontal cortex", "left pre-subgenual frontal cortex",
    "right superior temporal gyrus anterior part", "left superior temporal gyrus anterior part",
    "right supramarginal gyrus", "left supramarginal gyrus",
    "right insula anterior short gyrus", "left insula anterior short gyrus",
    "right insula middle short gyrus", "left insula middle short gyrus",
    "right insula posterior short gyrus", "left insula posterior short gyrus",
    "right insula anterior inferior cortex", "left insula anterior inferior cortex",
    "right insula anterior long gyrus", "left insula anterior long gyrus",
    "insula", "parietal lobe", "frontal lobe", "basal ganglia", "cingulate gyrus",
    "temporal lobe", "occipital lobe"
}

# Comprehensive mapping from private labels to SAT standard labels
BRAIN_LABEL_MAPPING = {
    # Thalamus variations
    "R Thalamus": "right thalamus",
    "L Thalamus": "left thalamus",
    "Right Thalamus": "right thalamus", 
    "Left Thalamus": "left thalamus",
    "Thalamus_R": "right thalamus",
    "Thalamus_L": "left thalamus",
    "R_Thalamus": "right thalamus",
    "L_Thalamus": "left thalamus",
    "Thalamus R": "right thalamus",
    "Thalamus L": "left thalamus",
    "Thalamus": "thalamus",
    
    # Hippocampus variations
    "R Hippocampus": "right hippocampus",
    "L Hippocampus": "left hippocampus",
    "Right Hippocampus": "right hippocampus",
    "Left Hippocampus": "left hippocampus",
    "Hippocampus_R": "right hippocampus",
    "Hippocampus_L": "left hippocampus",
    "R_Hippocampus": "right hippocampus",
    "L_Hippocampus": "left hippocampus",
    "Hippocampus R": "right hippocampus",
    "Hippocampus L": "left hippocampus",
    "Hippocampus": "hippocampus",
    
    # Caudate nucleus variations
    "R Caudate": "right caudate nucleus",
    "L Caudate": "left caudate nucleus",
    "Right Caudate": "right caudate nucleus",
    "Left Caudate": "left caudate nucleus",
    "Caudate_R": "right caudate nucleus",
    "Caudate_L": "left caudate nucleus",
    "R_Caudate": "right caudate nucleus",
    "L_Caudate": "left caudate nucleus",
    "Caudate R": "right caudate nucleus",
    "Caudate L": "left caudate nucleus",
    "Caudate Nucleus R": "right caudate nucleus",
    "Caudate Nucleus L": "left caudate nucleus",
    "R Caudate Nucleus": "right caudate nucleus",
    "L Caudate Nucleus": "left caudate nucleus",
    "Caudate": "caudate",
    
    # Putamen variations
    "R Putamen": "right putamen",
    "L Putamen": "left putamen",
    "Right Putamen": "right putamen",
    "Left Putamen": "left putamen",
    "Putamen_R": "right putamen",
    "Putamen_L": "left putamen",
    "R_Putamen": "right putamen",
    "L_Putamen": "left putamen",
    "Putamen R": "right putamen",
    "Putamen L": "left putamen",
    "Putamen": "putamen",
    
    # Amygdala variations
    "R Amygdala": "right amygdala",
    "L Amygdala": "left amygdala",
    "Right Amygdala": "right amygdala",
    "Left Amygdala": "left amygdala",
    "Amygdala_R": "right amygdala",
    "Amygdala_L": "left amygdala",
    "R_Amygdala": "right amygdala",
    "L_Amygdala": "left amygdala",
    "Amygdala R": "right amygdala",
    "Amygdala L": "left amygdala",
    "Amygdala": "amygdala",

    # Accumbens variations
    "R Accumbens": "right nucleus accumbens",
    "L Accumbens": "left nucleus accumbens",
    "Right Accumbens": "right nucleus accumbens",
    "Left Accumbens": "left nucleus accumbens",
    "Accumbens_R": "right nucleus accumbens",
    "Accumbens_L": "left nucleus accumbens",
    "R_Accumbens": "right nucleus accumbens",
    "L_Accumbens": "left nucleus accumbens",
    "Accumbens R": "right nucleus accumbens",
    "Accumbens L": "left nucleus accumbens",
    "Accumbens": "accumbens",
    
    # Pallidum/Globus Pallidus variations
    "R Pallidum": "right pallidum",
    "L Pallidum": "left pallidum",
    "Right Pallidum": "right pallidum",
    "Left Pallidum": "left pallidum",
    "R Globus Pallidus": "right pallidum",
    "L Globus Pallidus": "left pallidum",
    "Globus Pallidus R": "right pallidum",
    "Globus Pallidus L": "left pallidum",
    "Pallidus": "pallidum",
    "Pallidum": "pallidum",
    
    # Brainstem variations
    "Brainstem": "brainstem",
    "Brain Stem": "brainstem",
    "Brain_Stem": "brainstem",
    
    # Cerebellum variations
    "R Cerebellum": "right cerebellum",
    "L Cerebellum": "left cerebellum",
    "Right Cerebellum": "right cerebellum",
    "Left Cerebellum": "left cerebellum",
    "Cerebellum_R": "right cerebellum",
    "Cerebellum_L": "left cerebellum",
    "Cerebellum": "cerebellum",
    
    # Corpus Callosum variations
    "Corpus Callosum": "corpus callosum",
    "Corpus_Callosum": "corpus callosum",
    "CC": "corpus callosum",
    
    # Ventricle variations
    "R Lateral Ventricle": "right Lateral ventricle excluding temporal horn",
    "L Lateral Ventricle": "left Lateral ventricle excluding temporal horn",
    "Right Lateral Ventricle": "right Lateral ventricle excluding temporal horn",
    "Left Lateral Ventricle": "left Lateral ventricle excluding temporal horn",
    "Lateral Ventricle R": "right Lateral ventricle excluding temporal horn",
    "Lateral Ventricle L": "left Lateral ventricle excluding temporal horn",
    "3rd Ventricle": "Third ventricle",
    "Third Ventricle": "Third ventricle",
    "Ventricle 3": "Third ventricle",
}

def standardize_brain_labels(original_labels: List[str]) -> List[str]:
    """
    Standardize private brain region labels to SAT format
    
    Args:
        original_labels: List of original brain region labels
        
    Returns:
        List of standardized labels matching SAT Brain_Atlas format
    """
    standardized = []
    unmapped_labels = []
    
    for label in original_labels:
        # Direct mapping lookup
        if label in BRAIN_LABEL_MAPPING:
            standardized.append(BRAIN_LABEL_MAPPING[label])
        else:
            # Try basic preprocessing and remapping
            processed_label = _basic_label_processing(label)
            if processed_label in BRAIN_LABEL_MAPPING:
                standardized.append(BRAIN_LABEL_MAPPING[processed_label])
            elif processed_label.lower() in SAT_BRAIN_LABELS:
                standardized.append(processed_label.lower())
            else:
                # Keep original but warn
                standardized.append(label.lower())
                unmapped_labels.append(label)
                
    if unmapped_labels:
        warnings.warn(f"Unmapped brain labels found: {unmapped_labels}. "
                     f"Please add them to BRAIN_LABEL_MAPPING or verify they exist in SAT terminology.")
    
    return standardized

def _basic_label_processing(label: str) -> str:
    """Basic preprocessing for brain labels"""
    # Remove extra spaces and standardize format
    label = label.strip()
    label = " ".join(label.split())  # normalize whitespace
    
    # Common abbreviation expansions
    replacements = {
        " R ": " Right ",
        " L ": " Left ",
        "_R_": "_Right_",
        "_L_": "_Left_",
        " R": " Right",
        " L": " Left",
    }
    
    for old, new in replacements.items():
        if label.endswith(old.strip()):
            label = label.replace(old.strip(), new.strip())
        else:
            label = label.replace(old, new)
            
    return label

def validate_brain_mapping(labels: List[str]) -> Dict[str, List[str]]:
    """
    Validate brain label mapping completeness
    
    Args:
        labels: List of labels to validate
        
    Returns:
        Dict with 'mapped', 'unmapped', and 'invalid' label lists
    """
    standardized = standardize_brain_labels(labels)
    
    mapped = []
    unmapped = []
    invalid = []
    
    for orig, std in zip(labels, standardized):
        if std in SAT_BRAIN_LABELS:
            mapped.append(f"{orig} -> {std}")
        elif orig in BRAIN_LABEL_MAPPING:
            mapped.append(f"{orig} -> {std}")
        elif orig.lower() == std:
            unmapped.append(orig)
        else:
            invalid.append(f"{orig} -> {std} (not in SAT terminology)")
            
    return {
        'mapped': mapped,
        'unmapped': unmapped, 
        'invalid': invalid
    }

def get_brain_mapping_stats():
    """Get statistics about the brain label mapping"""
    print(f"Total SAT brain labels: {len(SAT_BRAIN_LABELS)}")
    print(f"Total mapping rules: {len(BRAIN_LABEL_MAPPING)}")
    print(f"Sample mappings:")
    for i, (k, v) in enumerate(list(BRAIN_LABEL_MAPPING.items())[:5]):
        print(f"  {k} -> {v}")
    print("  ...")

if __name__ == "__main__":
    # Example usage and testing
    test_labels = [
        "R Thalamus", "L Thalamus", "R Hippocampus", "L Hippocampus",
        "R Caudate", "L Caudate", "Brainstem", "Cerebellum"
    ]
    
    print("Testing brain label standardization:")
    print(f"Original: {test_labels}")
    standardized = standardize_brain_labels(test_labels)
    print(f"Standardized: {standardized}")
    
    print("\nValidation results:")
    validation = validate_brain_mapping(test_labels)
    for category, items in validation.items():
        print(f"{category}: {items}")
        
    print("\nMapping statistics:")
    get_brain_mapping_stats()