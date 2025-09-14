import argparse

def str2bool(v):
    return v.lower() in ('true', 't')
    
def parse_args_mapseg():
    """
    Parse arguments for MAPSeg_MAE inference and evaluation
    """
    parser = argparse.ArgumentParser(description='SAT with MAPSeg_MAE Inference/Evaluation')
    
    # Basic settings
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--gpu', type=str, default=None, help="Set which gpu to use")
    
    # Dataset and data loading
    parser.add_argument('--datasets_jsonl', type=str, required=True, help="JSONL file with inference data")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=str2bool, default=False)
    
    # Model configuration
    parser.add_argument('--crop_size', nargs='+', type=int, default=[288, 288, 96])
    parser.add_argument('--patch_size', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('--vision_backbone', type=str, default='UNET', help="vision backbone")
    parser.add_argument('--deep_supervision', type=bool, default=False, help="deep supervision")
    parser.add_argument('--online_crop', type=str2bool, default=False, help="online cropping")
    
    # Inference parameters
    parser.add_argument('--batchsize_3d', type=int, default=1, help="3d batch size for inference")
    parser.add_argument('--max_queries', type=int, default=256, help="maximum queries per batch")
    
    # Model checkpoints
    parser.add_argument('--checkpoint', type=str, required=True, help="SAT model checkpoint path")
    parser.add_argument('--resume', type=str2bool, default=False, help="resume from checkpoint")
    parser.add_argument('--partial_load', type=bool, default=True, help="partial load checkpoint")
    parser.add_argument('--unet_checkpoint', type=str, default=None, help="checkpoint path to resume")
    parser.add_argument('--cfg_file', type=str, default=None, help="cfg file for MPLSeg")
    
    # Text encoder
    parser.add_argument('--text_encoder', type=str, default='ours', help="text encoder type")
    parser.add_argument('--text_encoder_checkpoint', type=str, required=True, help="text encoder checkpoint")
    parser.add_argument('--text_encoder_partial_load', type=bool, default=False)
    parser.add_argument('--open_bert_layer', type=int, default=12)
    parser.add_argument('--open_modality_embed', type=bool, default=True)
    
    # Output settings
    parser.add_argument('--rcd_dir', type=str, required=True, help="results directory")
    parser.add_argument('--rcd_file', type=str, default=None, help="results file name")
    parser.add_argument('--visualization', type=bool, default=True, help="save visualization")
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100
    )
    
    # Evaluation metrics
    parser.add_argument('--dice', type=bool, default=True, help="compute Dice score")
    parser.add_argument('--nsd', type=bool, default=False, help="compute Normalized Surface Distance")
    parser.add_argument('--hausdorff', type=bool, default=False, help="compute Hausdorff distance")
    
    # MAPSeg_MAE specific arguments
    parser.add_argument('--mapseg_pretrained_path', type=str, default=None,
                       help="Path to MAPSeg_MAE pretrained weights (if not loaded in backbone)")
    parser.add_argument('--mapseg_embed_dim', type=int, default=512,
                       help="Embedding dimension for MAPSeg_MAE backbone")
    
    # Advanced inference options
    parser.add_argument('--test_time_adaptation', type=bool, default=False,
                       help="Enable test-time adaptation")
    parser.add_argument('--ensemble_inference', type=bool, default=False,
                       help="Use ensemble inference with multiple crops")
    parser.add_argument('--multi_scale_inference', type=bool, default=False,
                       help="Use multi-scale inference")
    
    # Output format options
    parser.add_argument('--save_individual_masks', type=bool, default=True,
                       help="Save individual organ masks")
    parser.add_argument('--save_combined_mask', type=bool, default=True,
                       help="Save combined segmentation mask")
    parser.add_argument('--output_format', type=str, default='nifti', choices=['nifti', 'numpy'],
                       help="Output format for segmentation masks")
    
    args = parser.parse_args()
    
    return args


def parse_args():
    """Backward compatibility - redirects to MAPSeg version"""
    return parse_args_mapseg()