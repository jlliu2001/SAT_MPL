import torch
import torch.nn as nn
import time
import os
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from .maskformer_mapseg import Maskformer_MAPSeg,Maskformer_MAPSegV2,Maskformer_MPLSeg  

from train.dist import is_master


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def build_maskformer_mapseg(args, device, gpu_id):
    """
    Build Maskformer with MAPSeg_MAE support
    """
    # Extract MAPSeg specific arguments
    mapseg_pretrained_path = getattr(args, 'checkpoint', None)
    mapseg_embed_dim = getattr(args, 'mapseg_embed_dim', 512)
    
    model = Maskformer_MAPSeg(
        vision_backbone=args.vision_backbone, 
        image_size=args.crop_size, 
        patch_size=args.patch_size, 
        deep_supervision=args.deep_supervision,
        mapseg_pretrained_path=mapseg_pretrained_path,
        mapseg_embed_dim=mapseg_embed_dim
    )

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    if is_master():
        print(f"** MODEL ** {get_parameter_number(model)['Total']/1e6}M parameters")
        if args.vision_backbone == 'MAPSeg_MAE':
            print(f"** MAPSeg_MAE ** Using embed_dim={mapseg_embed_dim}")
            if mapseg_pretrained_path:
                print(f"** MAPSeg_MAE ** Pretrained weights from: {mapseg_pretrained_path}")
            
    return model


def build_maskformer_mapsegV2(args, device, gpu_id):
    """
    Build Maskformer with MAPSeg_MAE support
    """
    # Extract MAPSeg specific arguments
    mapseg_pretrained_path = getattr(args, 'checkpoint', None)
    mapseg_embed_dim = getattr(args, 'mapseg_embed_dim', 512)
    
    model = Maskformer_MAPSegV2(
        vision_backbone=args.vision_backbone, 
        image_size=args.crop_size, 
        patch_size=args.patch_size, 
        deep_supervision=args.deep_supervision,
        mapseg_pretrained_path=mapseg_pretrained_path,
        mapseg_embed_dim=mapseg_embed_dim
    )

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    if is_master():
        print(f"** MODEL ** {get_parameter_number(model)['Total']/1e6}M parameters")
        if args.vision_backbone == 'MAPSeg_MAE':
            print(f"** MAPSeg_MAE ** Using embed_dim={mapseg_embed_dim}")
            if mapseg_pretrained_path:
                print(f"** MAPSeg_MAE ** Pretrained weights from: {mapseg_pretrained_path}")
            
    return model

def build_maskformer_MPLseg(args, device, gpu_id):
    """
    Build Maskformer with MAPSeg_MAE support
    """
    # Extract MAPSeg specific arguments
    mapseg_pretrained_path = getattr(args, 'checkpoint', None)
    mapseg_embed_dim = getattr(args, 'mapseg_embed_dim', 512)
    
    model = Maskformer_MPLSeg(
        vision_backbone=args.vision_backbone, 
        image_size=args.patch_size, 
        patch_size=[32, 32, 32], 
        deep_supervision=args.deep_supervision,
        mapseg_pretrained_path=mapseg_pretrained_path,
        mapseg_embed_dim=mapseg_embed_dim,
        cfg_file=args.cfg_file
    )

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    if is_master():
        print(f"** MODEL ** {get_parameter_number(model)['Total']/1e6}M parameters")
        if args.vision_backbone == 'MAPSeg_MAE':
            print(f"** MAPSeg_MAE ** Using embed_dim={mapseg_embed_dim}")
            if mapseg_pretrained_path:
                print(f"** MAPSeg_MAE ** Pretrained weights from: {mapseg_pretrained_path}")
            
    return model


def load_checkpoint_mapseg(checkpoint, 
                          resume, 
                          partial_load, 
                          model, 
                          device,
                          optimizer=None,
                          ):
    """
    Load checkpoint with MAPSeg_MAE support
    """
    
    if is_master():
        print('** CHECKPOINT ** : Load checkpoint from %s' % (checkpoint))
    
    
        
    # load part of the checkpoint
    if partial_load:
        
        model_dict =  model.state_dict()
        #model.load_pretrained_weights(checkpoint)
        # check difference
        checkpoint = torch.load(checkpoint, map_location=device)
        unexpected_state_dict = [k for k in checkpoint['model_state_dict'].keys() if k not in model_dict.keys()]
        missing_state_dict = [k for k in model_dict.keys() if k not in checkpoint['model_state_dict'].keys()]
        unmatchd_state_dict = [k for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
        # load partial parameters
        state_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
        #model_dict.update(state_dict)
        #model.load_state_dict(state_dict)
        model.load_state_dict(model_dict)
        # print('module.backbone.mae_cnn.encoders.0.block.conv1.groupnorm.weight:',model.state_dict()['module.backbone.mae_cnn.encoders.0.block.conv1.groupnorm.weight'])
        # print('local_encoder.0.basic_module.conv1.groupnorm.weight:',checkpoint['local_encoder.0.basic_module.conv1.groupnorm.weight'])
        
        if is_master():
            print('The following parameters are unexpected in SAT checkpoint:\n', unexpected_state_dict)
            print('The following parameters are missing in SAT checkpoint:\n', missing_state_dict)
            print('The following parameters have different shapes in SAT checkpoint:\n', unmatchd_state_dict)
            print('The following parameters are loaded in SAT:\n', state_dict.keys())
    else:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # if resume, load optimizer and step
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = int(checkpoint['step']) + 1
    else:
        start_step = 1
        
    return model, optimizer, start_step


def load_checkpoint_mapsegV2(mapseg_checkpoint, 
                          unet_checkpoint,
                          resume, 
                          partial_load, 
                          model, 
                          device,
                          optimizer=None,
                          ):
    """
    Load checkpoint with dual support for MAPSeg_MAE and UNET pretrained models
    
    Args:
        mapseg_checkpoint: path to MAPSeg pretrained weights (for vision backbone)
        unet_checkpoint: path to UNET-based SAT pretrained weights (for TransformerDecoder)
        resume: whether to resume training
        partial_load: whether to do partial loading
        model: the model to load weights into
        device: torch device
        optimizer: optimizer (optional)
    
    Returns:
        model, optimizer, start_step
    """
    
    if is_master():
        print('** DUAL CHECKPOINT LOADING **')
        print(f'MAPSeg checkpoint: {mapseg_checkpoint}')
        print(f'UNET checkpoint  : {unet_checkpoint}')
    
    model_dict = model.state_dict()
    loaded_params = []
    
    # 1. Load MAPSeg pretrained weights for vision backbone
    if mapseg_checkpoint is not None:
        if is_master():
            print('** Loading MAPSeg backbone weights **')
        
        try:
            mapseg_ckpt = torch.load(mapseg_checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if 'state_dict' in mapseg_ckpt:
                mapseg_state_dict = mapseg_ckpt['state_dict']
            elif 'model' in mapseg_ckpt:
                mapseg_state_dict = mapseg_ckpt['model']
            elif 'model_state_dict' in mapseg_ckpt:
                mapseg_state_dict = mapseg_ckpt['model_state_dict']
            else:
                mapseg_state_dict = mapseg_ckpt
            
            # Map MAPSeg weights to backbone structure (now using MAE_CNN_Original)
            # mapseg_loaded = {}
            # for key, value in mapseg_state_dict.items():
            #     # Remove 'module.' prefix if present
            #     if key.startswith('module.'):
            #         key = key[7:]
                
            #     # Map to MAE_CNN_Original backbone structure: module.backbone.mae_cnn.*
            #     new_key = f'module.backbone.mae_cnn.{key}'
                
            #     if new_key in model_dict and value.shape == model_dict[new_key].shape:
            #         mapseg_loaded[new_key] = value
            #         loaded_params.append(new_key)
            
            # model_dict.update(mapseg_loaded)
            # print('model_dict.keys():',model_dict.keys())

            if partial_load:
        
                # model_dict =  model.state_dict()
                #model.load_pretrained_weights(checkpoint)
                # check difference
                # checkpoint = torch.load(checkpoint, map_location=device)
                # unexpected_state_dict = [k for k in mapseg_state_dict.keys() if k not in model_dict.keys()]
                # missing_state_dict = [k for k in model_dict.keys() if k not in mapseg_state_dict.keys()]
                # unmatchd_state_dict = [k for k,v in mapseg_state_dict.items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
                # # load partial parameters
                # state_dict = {k:v for k,v in mapseg_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
                # loaded_params=[k for k,v in mapseg_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape]
                # model_dict.update(state_dict)
                #model.load_state_dict(state_dict)
                # model.load_state_dict(model_dict)
                # print('module.backbone.mae_cnn.encoders.0.block.conv1.groupnorm.weight:',model.state_dict()['module.backbone.mae_cnn.encoders.0.block.conv1.groupnorm.weight'])
                # print('local_encoder.0.basic_module.conv1.groupnorm.weight:',mapseg_ckpt['local_encoder.0.basic_module.conv1.groupnorm.weight'])
                mapseg_loaded = {}
                mapseg_state_dict_new={}
                for key, value in mapseg_state_dict.items():
                    # Remove 'module.' prefix if present
                    # if key.startswith('module.'):
                    #     key = key[7:]
                    
                    # Map to MAE_CNN_Original backbone structure: module.backbone.mae_cnn.*
                    new_key = f'module.{key}'
                    mapseg_state_dict_new[new_key] = value
                    
                    if new_key in model_dict and value.shape == model_dict[new_key].shape:
                        mapseg_loaded[new_key] = value
                        loaded_params.append(new_key)

                unexpected_state_dict = [k for k in mapseg_state_dict_new.keys() if k not in model_dict.keys()]
                missing_state_dict = [k for k in model_dict.keys() if k not in mapseg_state_dict_new.keys()]
                unmatchd_state_dict = [k for k,v in mapseg_state_dict_new.items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
                # load partial parameters
                state_dict = {k:v for k,v in mapseg_state_dict_new.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
                
                
                model_dict.update(mapseg_loaded)


                if is_master():
                    print('The following parameters are unexpected in MAE checkpoint:\n', unexpected_state_dict)
                    print('The following parameters are missing in MAE checkpoint:\n', missing_state_dict)
                    print('The following parameters have different shapes in MAE checkpoint:\n', unmatchd_state_dict)
                    print('The following parameters are loaded in MAE:\n', state_dict.keys())
                    
            if is_master():
                print(f'Loaded {len(state_dict)} MAPSeg backbone parameters')
                        
        except Exception as e:
            if is_master():
                print(f'Warning: Failed to load MAPSeg weights: {e}')
            
    # 2. Load UNET-based SAT weights for TransformerDecoder and other components
    if unet_checkpoint is not None:
        if is_master():
            print('** Loading TransformerDecoder and other components from UNET checkpoint **')
        
        try:
            unet_ckpt = torch.load(unet_checkpoint, map_location=device)
            
            if 'model_state_dict' in unet_ckpt:
                unet_state_dict = unet_ckpt['model_state_dict']
            else:
                unet_state_dict = unet_ckpt
            
            # Define components to load from UNET checkpoint (transformer decoder and related)
            target_components = [
                'projection_layer',          # TransformerDecoder layers
                'transformer_decoder',      # TransformerDecoder layers
                'query_proj',              # Query projection
                'mask_embed_proj',         # Mask embedding projection
                'mid_mask_embed_proj',     # Deep supervision projections
                # Skip projection_layer as it's designed for different backbone features
            ]
            
            unet_loaded = {}
            for key, value in unet_state_dict.items():
                # Check if this parameter belongs to transformer decoder components
                for component in target_components:
                    if component in key:
                        # Load transformer decoder and related components if shapes match
                        if key in model_dict and value.shape == model_dict[key].shape:
                            unet_loaded[key] = value
                            loaded_params.append(key)
                        else:
                            if is_master() and key in model_dict:
                                print(f'Skipping {key} due to shape mismatch: {value.shape} vs {model_dict[key].shape}')
                        break
            
            model_dict.update(unet_loaded)
            
            if is_master():
                print(f'Loaded {len(unet_loaded)} TransformerDecoder and related parameters')
                print('The following parameters are loaded in TransformerDecoder:\n', unet_loaded.keys())
                
            # Handle optimizer and step from UNET checkpoint
            if resume and 'optimizer_state_dict' in unet_ckpt:
                if optimizer is not None:
                    optimizer.load_state_dict(unet_ckpt['optimizer_state_dict'])
                start_step = int(unet_ckpt.get('step', 0)) + 1
            else:
                start_step = 1
        
        except Exception as e:
            if is_master():
                print(f'Warning: Failed to load UNET checkpoint: {e}')
            start_step = 1
    else:
        start_step = 1
    
    # 3. Load the combined state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
    
    if is_master():
        print(f'** DUAL LOADING SUMMARY **')
        print(f'Total parameters loaded: {len(loaded_params)}')
        print(f'Missing keys: {len(missing_keys)}')
        print(f'Unexpected keys: {len(unexpected_keys)}')
        
        if len(loaded_params) > 0:
            print('Successfully loaded components:')
            component_counts = {}
            for param in loaded_params:
                for component in ['backbone', 'transformer_decoder', 'query_proj', 'mask_embed_proj', 'mid_mask_embed_proj']:
                    if component in param:
                        component_counts[component] = component_counts.get(component, 0) + 1
                        break
            for comp, count in component_counts.items():
                print(f'  - {comp}: {count} parameters')
        
        if len(missing_keys) > 0:
            print(f'Missing keys (will use random initialization): {missing_keys}')  # Show first 10
        
    return model, optimizer, start_step


def load_checkpoint_MPLseg(mapseg_checkpoint, 
                          unet_checkpoint,
                          resume, 
                          partial_load, 
                          model, 
                          device,
                          optimizer=None,
                          ):
    """
    Load checkpoint with dual support for MAPSeg_MAE and UNET pretrained models
    
    Args:
        mapseg_checkpoint: path to MAPSeg pretrained weights (for vision backbone)
        unet_checkpoint: path to UNET-based SAT pretrained weights (for TransformerDecoder)
        resume: whether to resume training
        partial_load: whether to do partial loading
        model: the model to load weights into
        device: torch device
        optimizer: optimizer (optional)
    
    Returns:
        model, optimizer, start_step
    """
    
    if is_master():
        print('** DUAL CHECKPOINT LOADING **')
        print(f'MAPSeg checkpoint: {mapseg_checkpoint}')
        print(f'UNET checkpoint  : {unet_checkpoint}')
    
    model_dict = model.state_dict()
    loaded_params = []
    
    # 1. Load MAPSeg pretrained weights for vision backbone
    if mapseg_checkpoint is not None:
        if is_master():
            print('** Loading MAPSeg backbone weights **')
        
        try:
            mapseg_ckpt = torch.load(mapseg_checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if 'state_dict' in mapseg_ckpt:
                mapseg_state_dict = mapseg_ckpt['state_dict']
            elif 'model' in mapseg_ckpt:
                mapseg_state_dict = mapseg_ckpt['model']
            elif 'model_state_dict' in mapseg_ckpt:
                mapseg_state_dict = mapseg_ckpt['model_state_dict']
            else:
                mapseg_state_dict = mapseg_ckpt
            

            if partial_load:
        
                # model_dict =  model.state_dict()
                #model.load_pretrained_weights(checkpoint)
                # check difference
                # checkpoint = torch.load(checkpoint, map_location=device)
                # unexpected_state_dict = [k for k in mapseg_state_dict.keys() if k not in model_dict.keys()]
                # missing_state_dict = [k for k in model_dict.keys() if k not in mapseg_state_dict.keys()]
                # unmatchd_state_dict = [k for k,v in mapseg_state_dict.items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
                # # load partial parameters
                # state_dict = {k:v for k,v in mapseg_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
                # loaded_params=[k for k,v in mapseg_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape]
                # model_dict.update(state_dict)
                #model.load_state_dict(state_dict)
                # model.load_state_dict(model_dict)
                # print('module.backbone.mae_cnn.encoders.0.block.conv1.groupnorm.weight:',model.state_dict()['module.backbone.mae_cnn.encoders.0.block.conv1.groupnorm.weight'])
                # print('local_encoder.0.basic_module.conv1.groupnorm.weight:',mapseg_ckpt['local_encoder.0.basic_module.conv1.groupnorm.weight'])
                mapseg_loaded = {}
                mapseg_state_dict_new={}
                for key, value in mapseg_state_dict.items():
                    # Remove 'module.' prefix if present
                    # if key.startswith('module.'):
                    #     key = key[7:]
                    if key.startswith('teacher'):
                        continue
                    # Map to MAE_CNN_Original backbone structure: module.backbone.mae_cnn.*
                    if unet_checkpoint is not None:
                        new_key = f'module.mpl.{key}'
                    else:
                        new_key = key
                    mapseg_state_dict_new[new_key] = value
                    
                    if new_key in model_dict and value.shape == model_dict[new_key].shape:
                        mapseg_loaded[new_key] = value
                        loaded_params.append(new_key)

                unexpected_state_dict = [k for k in mapseg_state_dict_new.keys() if k not in model_dict.keys()]
                missing_state_dict = [k for k in model_dict.keys() if k not in mapseg_state_dict_new.keys()]
                unmatchd_state_dict = [k for k,v in mapseg_state_dict_new.items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
                # load partial parameters
                state_dict = {k:v for k,v in mapseg_state_dict_new.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
                
                
                model_dict.update(mapseg_loaded)


                if is_master():
                    print('The following parameters are unexpected in MAE checkpoint:\n', unexpected_state_dict)
                    print('The following parameters are missing in MAE checkpoint:\n', missing_state_dict)
                    print('The following parameters have different shapes in MAE checkpoint:\n', unmatchd_state_dict)
                    print('The following parameters are loaded in MAE:\n', state_dict.keys())
                    
            if is_master():
                print(f'Loaded {len(state_dict)} MAPSeg backbone parameters')
                        
        except Exception as e:
            if is_master():
                print(f'Warning: Failed to load MAPSeg weights: {e}')
            
    # 2. Load UNET-based SAT weights for TransformerDecoder and other components
    if unet_checkpoint is not None:
        if is_master():
            print('** Loading TransformerDecoder and other components from UNET checkpoint **')
        
        try:
            unet_ckpt = torch.load(unet_checkpoint, map_location=device)
            
            if 'model_state_dict' in unet_ckpt:
                unet_state_dict = unet_ckpt['model_state_dict']
            else:
                unet_state_dict = unet_ckpt
            
            # Define components to load from UNET checkpoint (transformer decoder and related)
            target_components = [
                'projection_layer',          # TransformerDecoder layers
                'transformer_decoder',      # TransformerDecoder layers
                'query_proj',              # Query projection
                'mask_embed_proj',         # Mask embedding projection
                'mid_mask_embed_proj',     # Deep supervision projections
                # Skip projection_layer as it's designed for different backbone features
            ]
            
            unet_loaded = {}
            for key, value in unet_state_dict.items():
                # Check if this parameter belongs to transformer decoder components
                for component in target_components:
                    if component in key:
                        # Load transformer decoder and related components if shapes match
                        if key in model_dict and value.shape == model_dict[key].shape:
                            unet_loaded[key] = value
                            loaded_params.append(key)
                        else:
                            if is_master() and key in model_dict:
                                print(f'Skipping {key} due to shape mismatch: {value.shape} vs {model_dict[key].shape}')
                        break
            
            model_dict.update(unet_loaded)
            
            if is_master():
                print(f'Loaded {len(unet_loaded)} TransformerDecoder and related parameters')
                print('The following parameters are loaded in TransformerDecoder:\n', unet_loaded.keys())
                
            # Handle optimizer and step from UNET checkpoint
            if resume and 'optimizer_state_dict' in unet_ckpt:
                if optimizer is not None:
                    optimizer.load_state_dict(unet_ckpt['optimizer_state_dict'])
                start_step = int(unet_ckpt.get('step', 0)) + 1
            else:
                start_step = 1
        
        except Exception as e:
            if is_master():
                print(f'Warning: Failed to load UNET checkpoint: {e}')
            start_step = 1
    else:
        start_step = 1
    
    # 3. Load the combined state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
    
    if is_master():
        print(f'** DUAL LOADING SUMMARY **')
        print(f'Total parameters loaded: {len(loaded_params)}')
        print(f'Missing keys: {len(missing_keys)}')
        print(f'Unexpected keys: {len(unexpected_keys)}')
        
        if len(loaded_params) > 0:
            print('Successfully loaded components:')
            component_counts = {}
            for param in loaded_params:
                for component in ['backbone', 'transformer_decoder', 'query_proj', 'mask_embed_proj', 'mid_mask_embed_proj']:
                    if component in param:
                        component_counts[component] = component_counts.get(component, 0) + 1
                        break
            for comp, count in component_counts.items():
                print(f'  - {comp}: {count} parameters')
        
        if len(missing_keys) > 0:
            print(f'Missing keys (will use random initialization): {missing_keys}')  # Show first 10
        
    return model, optimizer, start_step


def inherit_knowledge_encoder(knowledge_encoder_checkpoint,
                              model,
                              device
                              ):
    """
    Inherit unet encoder and multiscale feature projection layer from knowledge encoder
    """
    # inherit unet encoder and multiscale feature projection layer from knowledge encoder
    checkpoint = torch.load(knowledge_encoder_checkpoint, map_location=device)
        
    model_dict =  model.state_dict()
    visual_encoder_state_dict = {k.replace('atlas_tower', 'backbone'):v for k,v in checkpoint['model_state_dict'].items() if 'atlas_tower.encoder' in k}    # encoder部分
    model_dict.update(visual_encoder_state_dict)
    proj_state_dict = {k.replace('atlas_tower.', ''):v for k,v in checkpoint['model_state_dict'].items() if 'atlas_tower.projection_layer' in k}    # projection layer部分
    model_dict.update(proj_state_dict)
    model.load_state_dict(model_dict)
    
    if is_master():
        print('** CHECKPOINT ** : Inherit pretrained unet encoder from %s' % (knowledge_encoder_checkpoint))
        print('The following parameters are loaded in SAT:\n', list(visual_encoder_state_dict.keys())+list(proj_state_dict.keys()))
        
    return model


def load_mapseg_pretrained(mapseg_pretrained_path, model, device, freeze_encoder=False):
    """
    Load MAPSeg_MAE pretrained weights specifically
    
    Args:
        mapseg_pretrained_path: path to MAPSeg_MAE pretrained weights
        model: the SAT model with MAPSeg_MAE backbone
        device: torch device
        freeze_encoder: whether to freeze the encoder after loading
    """
    if is_master():
        print(f'** MAPSeg Pretrained ** : Load MAPSeg_MAE weights from {mapseg_pretrained_path}')
    
    try:
        checkpoint = torch.load(mapseg_pretrained_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            mapseg_state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            mapseg_state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            mapseg_state_dict = checkpoint['model_state_dict']
        else:
            mapseg_state_dict = checkpoint
        
        # Get current model state dict
        model_dict = model.state_dict()
        
        # Map MAPSeg_MAE weights to SAT model structure
        adapted_state_dict = {}
        
        for key, value in mapseg_state_dict.items():
            # Remove 'module.' prefix if present (from DataParallel)
            if key.startswith('module.'):
                key = key[7:]
            
            # Map to the MAPSeg backbone in SAT structure
            # The MAPSeg backbone is at module.backbone.mae_cnn
            new_key = f'module.backbone.mae_cnn.{key}'
            
            # Check if this key exists in our model and has matching dimensions
            if new_key in model_dict and value.shape == model_dict[new_key].shape:
                adapted_state_dict[new_key] = value
        
        # Update model dict and load
        model_dict.update(adapted_state_dict)
        model.load_state_dict(model_dict, strict=False)
        
        if is_master():
            print(f'** MAPSeg Pretrained ** : Successfully loaded {len(adapted_state_dict)} parameters')
            print('Loaded parameters:', list(adapted_state_dict.keys())[:10])  # Show first 10 keys
            
            if freeze_encoder:
                # Freeze the MAPSeg encoder
                if hasattr(model.module, 'freeze_backbone_encoder'):
                    model.module.freeze_backbone_encoder()
                    print('** MAPSeg Pretrained ** : Encoder frozen for fine-tuning')
                
    except Exception as e:
        if is_master():
            print(f'** MAPSeg Pretrained ** : Failed to load pretrained weights: {e}')
            print('** MAPSeg Pretrained ** : Continuing with random initialization...')
    
    return model


def setup_differential_learning_rates(model, args):
    """
    Setup different learning rates for encoder vs decoder components
    
    Args:
        model: the SAT model
        args: training arguments with lr, backbone_lr_ratio attributes
        
    Returns:
        parameter groups for optimizer
    """
    base_lr = args.lr
    backbone_lr_ratio = getattr(args, 'backbone_lr_ratio', 0.1)  # Default: 10x smaller LR for backbone
    
    # Get parameter groups
    if hasattr(model.module, 'get_backbone_encoder_params'):
        backbone_params = list(model.module.get_backbone_encoder_params())
        decoder_params = list(model.module.get_decoder_params())
    else:
        # Fallback: all model parameters with same LR
        backbone_params = []
        decoder_params = list(model.parameters())
    
    param_groups = []
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': base_lr * backbone_lr_ratio,
            'name': 'backbone_encoder'
        })
        
    param_groups.append({
        'params': decoder_params,
        'lr': base_lr,
        'name': 'decoder'
    })
    
    if is_master():
        print(f'** Learning Rates ** : Backbone encoder LR = {base_lr * backbone_lr_ratio}, Decoder LR = {base_lr}')
        print(f'** Parameter Groups ** : Backbone encoder ({len(backbone_params)} params), Decoder ({len(decoder_params)} params)')
    
    return param_groups


def create_optimizer_mapseg(model, args):
    """
    Create optimizer with differential learning rates for MAPSeg_MAE
    """
    if getattr(args, 'use_differential_lr', False) and args.vision_backbone == 'MAPSeg_MAE':
        param_groups = setup_differential_learning_rates(model, args)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=getattr(args, 'weight_decay', 0.01))
    else:
        # Standard optimizer for all parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_decay', 0.01))
    
    return optimizer


# Backward compatibility functions
def build_maskformer(args, device, gpu_id):
    """Backward compatibility wrapper"""
    return build_maskformer_mapseg(args, device, gpu_id)


def load_checkpoint(checkpoint, resume, partial_load, model, device, optimizer=None):
    """Backward compatibility wrapper"""
    return load_checkpoint_mapseg(checkpoint, resume, partial_load, model, device, optimizer)