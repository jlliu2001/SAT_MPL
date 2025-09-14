import argparse

def str2bool(v):
    return v.lower() in ('true', 't')

def parse_args_mapseg():
    """
    Parse arguments for MAPSeg_MAE training
    Extends the original SAT arguments with MAPSeg specific options
    """
    parser = argparse.ArgumentParser(description='SAT with MAPSeg_MAE Training')
    
    # Original SAT arguments
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--gpu', type=str, default=None, help="Set which gpu to use")
    parser.add_argument('--pin_memory', type=str2bool, default=False)
    
    # dataset  
    parser.add_argument('--datasets_jsonl', type=str)
    parser.add_argument('--dataset_config', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--allow_repeat', type=bool, default=True, help="allow repeat training with same data")
    
    # general
    parser.add_argument('--crop_size', nargs='+', type=int, default=[256, 256, 64])
    parser.add_argument('--patch_size', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('--vision_backbone', type=str, default='UNET', help="vision backbone")
    parser.add_argument('--deep_supervision', type=str2bool, default=False, help="deep supervision")
    
    # training parameters
    parser.add_argument('--step_num', type=int,nargs='+', default=200000, help="total training steps")
    parser.add_argument('--warmup', type=int,nargs='+', default=20000, help="warmup steps")
    parser.add_argument('--lr', nargs='+', type=float, default=[1e-4], help="learning rate")
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # batch and accumulation
    parser.add_argument('--batchsize_3d', type=int, default=1, help="3d batch size")
    parser.add_argument('--accumulate_grad_interval', type=int, default=1, help="gradient accumulation")
    parser.add_argument('--max_queries', type=int, default=32, help="maximum queries per batch")
    
    # checkpointing and logging
    parser.add_argument('--checkpoint', type=str, default=None, help="checkpoint path to resume")
    parser.add_argument('--resume', type=bool, default=False, help="resume from checkpoint")
    parser.add_argument('--partial_load', type=bool, default=False, help="partial load checkpoint")
    parser.add_argument('--save_large_interval', type=int, default=10000, help="large save interval")
    parser.add_argument('--save_small_interval', type=int, default=1000, help="small save interval")
    parser.add_argument('--log_step_interval', type=int, default=1000, help="log interval")
    parser.add_argument('--log_dir', type=str, default='log', help="log directory")
    parser.add_argument('--name', type=str, default='sat_mapseg', help="experiment name")
    parser.add_argument('--cfg_file', type=str, default=None, help="cfg file for MPLSeg")
    
    # text encoder
    parser.add_argument('--text_encoder', type=str, default='ours', help="text encoder type")
    parser.add_argument('--text_encoder_checkpoint', type=str, help="text encoder checkpoint")
    parser.add_argument('--text_encoder_partial_load', type=bool, default=False)
    parser.add_argument('--open_bert_layer', type=int, default=12)
    parser.add_argument('--open_modality_embed', type=bool, default=True)
    
    # knowledge encoder inheritance (for traditional SAT)
    parser.add_argument('--inherit_knowledge_encoder', type=str, default=None, 
                       help="inherit from knowledge encoder checkpoint")
    
    # MAPSeg_MAE specific arguments
    parser.add_argument('--mapseg_pretrained_path', type=str, default=None,
                       help="Path to MAPSeg_MAE pretrained weights")
    parser.add_argument('--mapseg_embed_dim', type=int, default=512,
                       help="Embedding dimension for MAPSeg_MAE backbone")
    parser.add_argument('--mapseg_load_separately', type=bool, default=False,
                       help="Load MAPSeg weights separately (not in backbone init)")
    parser.add_argument('--mapseg_freeze_encoder', type=bool, default=False,
                       help="Freeze MAPSeg encoder initially")
    
    # Dual checkpoint loading for MAPSegV2
    parser.add_argument('--unet_checkpoint', type=str, default=None,
                       help="Path to UNET-based SAT weights for transformer decoder")
    
    # Parameter freezing for fine-tuning
    parser.add_argument('--freeze_encoder', type=bool, default=True,
                       help="Freeze MAPSeg encoder during fine-tuning")
    parser.add_argument('--freeze_text_encoder', type=bool, default=True,
                       help="Freeze text encoder during fine-tuning")
    
    # Differential learning rates
    parser.add_argument('--use_differential_lr', type=bool, default=False,
                       help="Use different learning rates for backbone vs decoder")
    parser.add_argument('--backbone_lr_ratio', type=float, default=0.1,
                       help="Ratio of backbone LR to base LR")
    
    # Progressive unfreezing
    parser.add_argument('--progressive_unfreeze', type=bool, default=False,
                       help="Progressively unfreeze backbone during training")
    parser.add_argument('--unfreeze_steps', nargs='+', type=int, default=None,
                       help="Steps at which to unfreeze backbone components")
    
    # Feature alignment
    parser.add_argument('--feature_alignment_warmup', type=bool, default=False,
                       help="Add feature alignment loss during warmup")
    parser.add_argument('--alignment_warmup_steps', type=int, default=None,
                       help="Steps for feature alignment warmup")
    parser.add_argument('--alignment_loss_weight', type=float, default=0.1,
                       help="Weight for feature alignment loss")
    
    # Advanced MAPSeg training options
    parser.add_argument('--mapseg_mae_loss_weight', type=float, default=0.0,
                       help="Weight for MAE reconstruction loss (0 to disable)")
    parser.add_argument('--mapseg_contrastive_loss', type=bool, default=False,
                       help="Add contrastive loss between text and vision features")
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.05,
                       help="Weight for contrastive loss")
    
    # Semantic alignment fine-tuning options
    parser.add_argument('--alignment_finetune', type=bool, default=False,
                       help="Enable semantic alignment fine-tuning mode")
    
    args = parser.parse_args()
    
    # Post-process arguments
    if args.unfreeze_steps is None and args.progressive_unfreeze:
        # Default unfreezing schedule: 1/3 and 2/3 of training
        args.unfreeze_steps = [args.step_num // 3, args.step_num * 2 // 3]
    
    if args.alignment_warmup_steps is None and args.feature_alignment_warmup:
        # Default: same as regular warmup
        args.alignment_warmup_steps = args.warmup
    
    return args


def parse_args():
    """Backward compatibility - redirects to MAPSeg version"""
    return parse_args_mapseg()