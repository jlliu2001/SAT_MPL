"""
Private Brain Data Preprocessing for SAT Framework
Converts private brain segmentation data to SAT-compatible JSONL format
Handles directory structure: subject_folder/AA_1.nii, AA_1_seg.nii, AA_1_seg_Label.txt
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import warnings

# Import the brain region mapping
try:
    from utils.brain_region_mapping import standardize_brain_labels, validate_brain_mapping
except ImportError:
    print("Warning: Cannot import brain_region_mapping. Please ensure utils/brain_region_mapping.py exists.")
    def standardize_brain_labels(labels): return labels
    def validate_brain_mapping(labels): return {'mapped': [], 'unmapped': labels, 'invalid': []}

def read_label_file(label_file_path: str) -> List[str]:
    """
    Read labels from label text file
    Supports various formats: one label per line, comma-separated, etc.
    """
    labels = []
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Try different parsing methods
        if '\n' in content:
            # One label per line
            labels = [line.strip() for line in content.split('\n') if line.strip()]
        elif ',' in content:
            # Comma-separated
            labels = [label.strip() for label in content.split(',') if label.strip()]
        else:
            # Single label or space-separated
            labels = [label.strip() for label in content.split() if label.strip()]
            
    except Exception as e:
        warnings.warn(f"Error reading label file {label_file_path}: {e}")
        
    return labels

def detect_modality(image_path: str) -> str:
    """
    Detect modality from image file (simple heuristic)
    You may need to implement more sophisticated detection based on your data
    """
    # Default to MRI for brain data, but you can customize this
    # Could check image intensity statistics, file naming patterns, etc.
    return "MRI"

def find_data_triplets(root_path: str) -> List[Dict]:
    """
    Find all data triplets (image, mask, labels) in the directory structure
    Expected structure: root_path/subject_folder/subject_id.nii, subject_id_seg.nii, subject_id_seg_Label.txt
    """
    root_path = Path(root_path)
    data_triplets = []
    
    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")
    
    # Scan all subdirectories
    for subject_dir in root_path.iterdir():
        if not subject_dir.is_dir():
            continue
            
        subject_name = subject_dir.name
        print(f"Processing subject directory: {subject_name}")
        
        # Find all potential image files in this subject directory
        # image_files = []
        # for ext in ['.nii', '.nii.gz']:
        #     pattern = f"*{ext}"
        #     potential_images = list(subject_dir.glob(pattern))
        #     # Filter out segmentation files
        #     image_files.extend([f for f in potential_images if not ('_seg' in f.name or '_mask' in f.name)])
        
        image_file = os.path.join(root_path,subject_name,subject_name,subject_name,subject_name+'_1_RAS.nii.gz')
        base_img_name=subject_name+'_1_RAS.nii.gz'
        # for image_file in image_files:
            # Extract base name (without extension)
        if image_file.endswith('.nii.gz'):
            base_name = image_file[:-11]  # Remove .nii.gz
        else:
            base_name = image_file[:-4]  # Remove .nii
            
        # Look for corresponding segmentation and label files
        seg_patterns = [f"{base_name}_seg_subcortical_RAS.nii", f"{base_name}_seg_subcortical_RAS.nii.gz", f"{base_name}_mask_subcortical_RAS.nii", f"{base_name}_mask_subcortical_RAS.nii.gz"]
        label_patterns = [f"{base_name}_seg_Label.txt", f"{base_name}_Label.txt", f"{base_name}_labels.txt"]
        
        seg_file = None
        label_file = None
        
        # Find segmentation file
        for pattern in seg_patterns:
            potential_seg = subject_dir / pattern
            if potential_seg.exists():
                seg_file = potential_seg
                break
                
        # Find label file  
        for pattern in label_patterns:
            potential_label = subject_dir / pattern
            if potential_label.exists():
                label_file = potential_label
                break
        
        if seg_file and label_file:
            data_triplets.append({
                'subject': subject_name,
                'base_name': base_img_name,
                'image': str(image_file),
                'mask': str(seg_file),
                'label_file': str(label_file)
            })
            print(f"  Found triplet: {base_name}")
        else:
            missing = []
            if not seg_file:
                missing.append("segmentation file")
            if not label_file:
                missing.append("label file")
            warnings.warn(f"Incomplete data for {image_file}: missing {', '.join(missing)}")
    
    return data_triplets

def process_private_brain_data(root_path: str, 
                             output_jsonl: str,
                             dataset_name: str = "PrivateBrainData",
                             modality: Optional[str] = None) -> None:
    """
    Process private brain data and generate JSONL file
    
    Args:
        root_path: Root directory containing subject folders
        output_jsonl: Output JSONL file path
        dataset_name: Name for the dataset
        modality: Force modality (if None, will auto-detect)
    """
    
    print(f"Processing private brain data from: {root_path}")
    print(f"Output JSONL: {output_jsonl}")
    
    # Find all data triplets
    triplets = find_data_triplets(root_path)
    
    if not triplets:
        raise ValueError("No valid data triplets found in the specified directory")
        
    print(f"Found {len(triplets)} data triplets")
    
    # Process each triplet
    jsonl_data = []
    all_original_labels = []
    all_standardized_labels = []
    
    for triplet in triplets:
        print(f"\nProcessing: {triplet['base_name']}")
        
        # Read labels from file
        original_labels = read_label_file(triplet['label_file'])
        if not original_labels:
            warnings.warn(f"No labels found in {triplet['label_file']}, skipping")
            continue
            
        print(f"  Original labels ({len(original_labels)}): {original_labels}")
        
        # Standardize labels
        standardized_labels = standardize_brain_labels(original_labels)
        print(f"  Standardized labels ({len(standardized_labels)}): {standardized_labels}")
        
        # Detect modality if not specified
        detected_modality = modality if modality else detect_modality(triplet['image'])
        
        # Create JSONL entry
        jsonl_entry = {
            'image': triplet['image'],
            'mask': triplet['mask'], 
            'label': standardized_labels,
            'modality': detected_modality,
            'dataset': dataset_name,
            'official_split': 'unknown',  # Default to train, can be modified later
            'patient_id': triplet['base_name']
        }
        
        jsonl_data.append(jsonl_entry)
        all_original_labels.extend(original_labels)
        all_standardized_labels.extend(standardized_labels)
    
    # Validate label mappings
    print(f"\n=== Label Mapping Validation ===")
    validation_result = validate_brain_mapping(list(set(all_original_labels)))
    
    print(f"Mapped labels ({len(validation_result['mapped'])})")
    for mapping in validation_result['mapped'][:10]:  # Show first 10
        print(f"  {mapping}")
    if len(validation_result['mapped']) > 10:
        print(f"  ... and {len(validation_result['mapped']) - 10} more")
        
    if validation_result['unmapped']:
        print(f"\nUnmapped labels ({len(validation_result['unmapped'])}):")
        for label in validation_result['unmapped']:
            print(f"  {label}")
        warnings.warn("Some labels are unmapped. Consider adding them to BRAIN_LABEL_MAPPING.")
        
    if validation_result['invalid']:
        print(f"\nInvalid labels ({len(validation_result['invalid'])}):")
        for label in validation_result['invalid']:
            print(f"  {label}")
        warnings.warn("Some labels don't match SAT terminology after mapping.")
    
    # Write JSONL file
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n=== Processing Complete ===")
    print(f"Generated JSONL with {len(jsonl_data)} entries")
    print(f"Output saved to: {output_jsonl}")
    print(f"Total unique labels: {len(set(all_standardized_labels))}")
    
    # Save label mapping report
    report_path = output_jsonl.replace('.jsonl', '_label_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Private Brain Data Label Mapping Report ===\n\n")
        f.write(f"Total subjects processed: {len(jsonl_data)}\n")
        f.write(f"Total unique original labels: {len(set(all_original_labels))}\n")
        f.write(f"Total unique standardized labels: {len(set(all_standardized_labels))}\n\n")
        
        f.write("=== Mapped Labels ===\n")
        for mapping in validation_result['mapped']:
            f.write(f"{mapping}\n")
            
        f.write("\n=== Unmapped Labels ===\n")
        for label in validation_result['unmapped']:
            f.write(f"{label}\n")
            
        f.write("\n=== Invalid Labels ===\n")
        for label in validation_result['invalid']:
            f.write(f"{label}\n")
            
    print(f"Label mapping report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Process private brain segmentation data for SAT training')
    parser.add_argument('--root_path', type=str, required=True,
                       help='Root directory containing subject folders')
    parser.add_argument('--output_jsonl', type=str, required=True,
                       help='Output JSONL file path')
    parser.add_argument('--dataset_name', type=str, default='PrivateBrainData',
                       help='Name for the dataset')
    parser.add_argument('--modality', type=str, choices=['CT', 'MRI', 'PET'],
                       help='Force modality (if not specified, will auto-detect)')
    
    args = parser.parse_args()
    
    try:
        process_private_brain_data(
            root_path=args.root_path,
            output_jsonl=args.output_jsonl, 
            dataset_name=args.dataset_name,
            modality=args.modality
        )
    except Exception as e:
        print(f"Error processing data: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    # Example usage:
    # python preprocess_private_brain_data.py \
    #   --root_path /path/to/private/brain/data \
    #   --output_jsonl private_brain_data.jsonl \
    #   --dataset_name MyBrainData \
    #   --modality MRI
    
    exit(main())