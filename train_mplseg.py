import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torch.distributed as dist

from data.build_dataset import build_dataset, build_dataset_MPL

from model.build_model_mapseg import (
    build_maskformer_mapseg, 
    build_maskformer_mapsegV2,
    build_maskformer_MPLseg,
    load_checkpoint_mapseg, 
    load_checkpoint_mapsegV2,
    load_checkpoint_MPLseg,
    inherit_knowledge_encoder,
    load_mapseg_pretrained,
    create_optimizer_mapseg
)
from model.text_encoder import Text_Encoder

from train.params_mapseg import parse_args_mapseg
from train.logger import set_up_log
from train.loss import BinaryDiceLoss
from train.scheduler import cosine_lr
from train.trainer import Trainer, Trainer_MPL
from train.dist import is_master
from train.loss_softmax import SoftmaxDiceLoss, SoftmaxCELoss

# ProtoAtlas imports (optional)
try:
    from proto_atlas_adapter import create_proto_atlas_adapter, create_proto_atlas_mse_adapter
    PROTO_ATLAS_AVAILABLE = True
except ImportError:
    PROTO_ATLAS_AVAILABLE = False
    print("Warning: proto_atlas_adapter not available. ProtoAtlas loss disabled.")

def set_seed(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # new seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def freeze_mapseg_encoder(model):
    """Freeze MAPSeg encoder parameters"""
    # if hasattr(model.module, 'local_encoder'):
    #     for param in model.module.local_encoder.parameters():
    #         param.requires_grad = False
    #     if is_master():
    #         print("** MAPSeg Training ** : Frozen local_encoder parameters")
    
    # Also freeze other encoder-related components if they exist
    encoder_components = ['mpl']
    for component_name in encoder_components:
        if hasattr(model.module, component_name):
            for param in getattr(model.module, component_name).parameters():
                param.requires_grad = False
            if is_master():
                print(f"** MAPSeg Training ** : Frozen {component_name} parameters")

def freeze_text_encoder(text_encoder):
    """Freeze text encoder parameters"""
    for param in text_encoder.parameters():
        param.requires_grad = False
    if is_master():
        print("** MAPSeg Training ** : Frozen text_encoder parameters")

def freeze_transformer_decoder(model):
    """Freeze text encoder parameters"""
    target_components = [
                'transformer_decoder',      # TransformerDecoder layers
                'query_proj',              # Query projection
                'mask_embed_proj',         # Mask embedding projection
                # Skip projection_layer as it's designed for different backbone features
            ]
    for component_name in target_components:
        if hasattr(model.module, component_name):
            for param in getattr(model.module, component_name).parameters():
                param.requires_grad = False
            if is_master():
                print(f"** MAPSeg Training ** : Frozen {component_name} parameters")



def get_trainable_parameters(model, text_encoder, args):
    """Get trainable parameters for MAPSeg fine-tuning"""
    trainable_params = []
    
    # Get model parameters (excluding frozen encoder parts)
    for name, param in model.named_parameters():
        # Skip encoder parameters if they should be frozen
        # if args.vision_backbone == 'MAPSeg_MAE' and getattr(args, 'freeze_encoder', True):
        #     if any(encoder_part in name for encoder_part in ['mpl']):
        #         continue
            # elif any(encoder_part in name for encoder_part in ['transformer_decoder','query_proj','mask_embed_proj']):
            #     continue
        trainable_params.append(param)
    
    # Get text encoder parameters (excluding if frozen)
    if not getattr(args, 'freeze_text_encoder', True):
        trainable_params.extend(list(text_encoder.parameters()))
    
    if is_master():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        print('******training params*******')
        for name, param in model.named_parameters():
            if param.requires_grad==True:
                print(f"** param {name} need to train ** :")
        print(f"** MAPSeg Training ** : {trainable_count}/{total_params} parameters are trainable")
    
    return trainable_params

def main():
    # get configs
    args = parse_args_mapseg()
    
    # set logger
    if is_master():
        checkpoint_dir, tb_writer, log_file = set_up_log(args)
    else:
        checkpoint_dir = None
        tb_writer = None
        log_file = None
        
    # set random seed for reproducibility
    # set_seed(args)
    
    # set up distribution (identify the device of current process)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    # display
    if is_master():
        print('** GPU NUM ** : ', torch.cuda.device_count())  # 打印gpu数量
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
        print(f'** MAPSeg Training ** : Using backbone {args.vision_backbone}')
        if args.vision_backbone == 'MAPSeg_MAE':
            print(f'** MAPSeg_MAE ** : embed_dim={getattr(args, "mapseg_embed_dim", 512)}')
    rank = dist.get_rank()
    print(f"** DDP ** : Start running DDP on rank {rank}.")
    
    # dataset and loader
    trainset, trainloader, sampler = build_dataset_MPL(args)
    
    # set model (by default gpu) - use MAPSegV2 for MAPSeg_MAE backbone
    if args.vision_backbone == 'MAPSeg_MAE':
        # model = build_maskformer_mapsegV2(args, device, gpu_id)
        model=build_maskformer_MPLseg(args, device, gpu_id)
    # else:
    #     model = build_maskformer_mapseg(args, device, gpu_id)
    
    # Load MAPSeg_MAE pretrained weights if specified
    if args.vision_backbone == 'MAPSeg_MAE' and hasattr(args, 'mapseg_pretrained_path') and args.mapseg_pretrained_path:
        freeze_encoder = getattr(args, 'mapseg_freeze_encoder', True)
        model = load_mapseg_pretrained(args.mapseg_pretrained_path, model, device, freeze_encoder)
    
    # build, load and set trainer parameters in knowledge encoder
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        partial_load=args.text_encoder_partial_load,
        open_bert_layer=args.open_bert_layer,
        open_modality_embed=args.open_modality_embed,
        gpu_id=gpu_id,
        device=device,
        # 新增：对于MAPSeg对齐微调，冻结text encoder
        freeze_for_alignment=getattr(args, 'freeze_text_encoder', True)
    )
    
    # Freeze encoder and text encoder for MAPSeg fine-tuning
    if args.vision_backbone == 'MAPSeg_MAE':
        # if getattr(args, 'freeze_encoder', True):
        #     freeze_mapseg_encoder(model)
        #     print('------freeze MAPseg------')
            # freeze_transformer_decoder(model)
        if getattr(args, 'freeze_text_encoder', True):
            freeze_text_encoder(text_encoder)
    
    # set loss calculator
    # dice_loss = BinaryDiceLoss(reduction='none')
    dice_loss = SoftmaxDiceLoss(reduction='none')
    bce_w_logits_loss = nn.BCEWithLogitsLoss(reduction='none') # safe for amp

    # ========================================
    # ProtoAtlas Loss Adapter (Optional)
    # ========================================
    proto_atlas_adapter = None
    if PROTO_ATLAS_AVAILABLE and getattr(args, 'use_proto_atlas', False):
        if is_master():
            print("\n" + "=" * 80)
            print("Initializing ProtoAtlas Loss Adapter")
            print("=" * 80)

        # Define all label names (must match pretraining)
        all_label_names = ["hippocampus", "amygdala", "caudate", "putamen",
                          "pallidum", "thalamus", "accumbens"]

        try:
            # proto_atlas_adapter = create_proto_atlas_adapter(
            #     shape_prior_path=args.shape_prior_path,
            #     shape_encoder_path=args.shape_encoder_path,
            #     loc_prior_path=args.loc_prior_path,
            #     loc_encoder_path=args.loc_encoder_path,
            #     all_label_names=all_label_names,
            #     device=device,
            #     lambda_atlas=getattr(args, 'lambda_atlas', 0.5),
            #     lambda_shape=getattr(args, 'lambda_shape', 0.5),
            #     lambda_loc=getattr(args, 'lambda_loc', 0.5),
            #     use_location_loss=getattr(args, 'use_location_loss', True),
            #     shape_emb_dim=getattr(args, 'shape_emb_dim', 128),
            #     loc_emb_dim=getattr(args, 'loc_emb_dim', 64),
            #     use_e3nn=getattr(args, 'use_e3nn', False)
            # )

            proto_atlas_adapter = create_proto_atlas_mse_adapter(
                shape_encoder_path=args.shape_encoder_path,
                loc_prior_path=args.loc_prior_path,
                loc_encoder_path=args.loc_encoder_path,
                all_label_names=all_label_names,
                device=device,
                lambda_atlas=getattr(args, 'lambda_atlas', 0.5),
                lambda_shape=getattr(args, 'lambda_shape', 0.5),
                lambda_loc=getattr(args, 'lambda_loc', 0.5),
                use_location_loss=getattr(args, 'use_location_loss', True),
                shape_emb_dim=getattr(args, 'shape_emb_dim', 128),
                loc_emb_dim=getattr(args, 'loc_emb_dim', 128),
                use_e3nn=getattr(args, 'use_e3nn', False)
            )
            

            if is_master():
                print("\n✓ ProtoAtlas adapter created successfully!")
                print(f"  lambda_atlas: {getattr(args, 'lambda_atlas', 0.5)}")
                print(f"  lambda_shape: {getattr(args, 'lambda_shape', 0.5)}")
                print(f"  lambda_loc: {getattr(args, 'lambda_loc', 0.5)}")
                print(f"  use_location_loss: {getattr(args, 'use_location_loss', True)}")

        except Exception as e:
            if is_master():
                print(f"\n✗ Failed to create ProtoAtlas adapter: {e}")
                print("  Continuing with standard training (no ProtoAtlas loss)")
            proto_atlas_adapter = None
    
    # set optimizer with differential learning rates for MAPSeg_MAE
    if args.vision_backbone == 'MAPSeg_MAE':
        # For MAPSeg fine-tuning, only use trainable parameters
        trainable_params = get_trainable_parameters(model, text_encoder, args)
        optimizer = optim.AdamW(
            trainable_params,
            lr=args.lr[0],
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=getattr(args, 'weight_decay', 0.01)
        )
    else:
        # Standard optimizer for all parameters
        target_parameters = list(model.parameters()) + list(text_encoder.parameters())
        optimizer = optim.AdamW(
            target_parameters,
            lr=args.lr[0],
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
    
    # set scheduler
    total_steps = args.step_num
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # if restart cosine annealing, total_steps = sum of steps in each stage
    if isinstance(total_steps, list):
        total_steps = sum(total_steps)
    
    # if restart cosine annealing, total_steps = sum of steps in each stage
    start_step = 1
    
    # load checkpoint - use appropriate loading function based on backbone
    if args.checkpoint is not None:
        if args.vision_backbone == 'MAPSeg_MAE':
            # For MAPSeg, use dual checkpoint loading
            unet_checkpoint = getattr(args, 'unet_checkpoint', None)
            model, optimizer, start_step = load_checkpoint_MPLseg(
                args.checkpoint,  # MAPSeg checkpoint
                unet_checkpoint,  # UNET checkpoint for transformer decoder
                args.resume, 
                args.partial_load,
                model,
                device,
                optimizer,
            )
        else:
            print('after fine-tuning')
            model, optimizer, start_step = load_checkpoint_mapseg(
                args.checkpoint, 
                args.resume, 
                args.partial_load,
                model,
                device,
                optimizer,
            )
    
    # inherit knowledge encoder (for traditional SAT models)
    if args.inherit_knowledge_encoder and args.vision_backbone != 'MAPSeg_MAE':
        model = inherit_knowledge_encoder(args.inherit_knowledge_encoder, model, device)
    
    # MAPSeg_MAE specific training schedule
    if args.vision_backbone == 'MAPSeg_MAE':
        # Progressive unfreezing schedule
        if hasattr(args, 'progressive_unfreeze') and args.progressive_unfreeze:
            unfreeze_steps = getattr(args, 'unfreeze_steps', [args.step_num // 3, args.step_num // 2])
            if is_master():
                print(f"** MAPSeg Training ** : Progressive unfreezing at steps: {unfreeze_steps}")
        
        # Feature alignment warmup
        if hasattr(args, 'feature_alignment_warmup') and args.feature_alignment_warmup:
            alignment_steps = getattr(args, 'alignment_warmup_steps', args.warmup)
            if is_master():
                print(f"** MAPSeg Training ** : Feature alignment warmup for {alignment_steps} steps")
    
    # start training
    # trainer = Trainer(
    #     args=args,
    #     model=model,
    #     text_encoder=text_encoder,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     trainloader=trainloader,
    #     sampler=sampler,
    #     device=device,
    #     dice_loss=dice_loss,
    #     bce_w_logits_loss=bce_w_logits_loss,
    #     start_step=start_step,
    #     log_params=(checkpoint_dir, tb_writer, log_file)
    # )

    trainer = Trainer_MPL(
                args=args,
                model=model,
                text_encoder=text_encoder,
                device=device,
                trainset=trainset,
                trainloader=trainloader,
                sampler=sampler,
                dice_loss=dice_loss,
                bce_w_logits_loss=bce_w_logits_loss,
                optimizer=optimizer,
                scheduler=scheduler,
                tb_writer=tb_writer,
                checkpoint_dir=checkpoint_dir,
                log_file=log_file,
                proto_atlas_adapter=proto_atlas_adapter  # Pass ProtoAtlas adapter (can be None)
                )
            #
    
    # MAPSeg specific training callbacks
    if args.vision_backbone == 'MAPSeg_MAE':
        # Add progressive unfreezing callback
        def progressive_unfreeze_callback(step):
            if hasattr(args, 'progressive_unfreeze') and args.progressive_unfreeze:
                unfreeze_steps = getattr(args, 'unfreeze_steps', [args.step_num // 3, args.step_num // 2])
                if step == unfreeze_steps[0]:
                    # Unfreeze encoder parameters
                    if hasattr(model.module, 'local_encoder'):
                        for param in model.module.local_encoder.parameters():
                            param.requires_grad = True
                        if is_master():
                            print(f"** Step {step} ** : Unfroze MAPSeg_MAE encoder")
        
        # Set the callback in trainer if supported
        if hasattr(trainer, 'add_step_callback'):
            trainer.add_step_callback(progressive_unfreeze_callback)

    for step in range(start_step, total_steps+1): 
        
        # make sure the train is not interrupted
        if is_master() and step%10==0:
            print(f'Training Step %d'%step)
        
        # accmulate grad
        for accum in range(args.accumulate_grad_interval):
            
            trainer.train_one_step(step,samples_per_epoch=20)
    
    # trainer.train()
    
    if is_master():
        print("Training completed successfully!")

if __name__ == '__main__':
    main()