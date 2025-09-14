import os
import datetime
import random
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import torch.distributed as dist

from data.evaluate_dataset_mpl import Evaluate_Dataset_MPL, Evaluate_Dataset_OnlineCrop_MPL, collate_fn_mpl
from model.build_model_mapseg import (
    build_maskformer_MPLseg,
    load_checkpoint_MPLseg
)
from model.text_encoder import Text_Encoder
from evaluate.evaluator_mpl import evaluate_mpl
from train.dist import is_master
from evaluate.params_mapseg import parse_args_mapseg

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

def main(args):
    # set gpu
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=10800))   # might takes a long time to sync between process
    
    # dispaly
    if is_master():
        print('** GPU NUM ** : ', torch.cuda.device_count())  # 打印gpu数量
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
        print(f'** MPL Evaluation ** : Using MPL-integrated SAT model')
    rank = dist.get_rank()
    print(f"** DDP ** : Start running DDP on rank {rank}.")
    
    # file to save the detailed metrics
    csv_path = f'{args.rcd_dir}/{args.rcd_file}.csv'
    txt_path = f'{args.rcd_dir}/{args.rcd_file}.txt'
    if is_master():
        Path(args.rcd_dir).mkdir(exist_ok=True, parents=True)
        print(f'Detailed Results will be Saved to {csv_path} and {txt_path}')
        
    # resume an evaluation if specified
    evaluated_samples = set()
    if args.resume:
        prefix = os.path.basename(csv_path).replace('.csv', '_tmp_rank')  # xxx/test/step_xxx.csv --> step_xxx_tmp_rank
        for file_name in os.listdir(args.rcd_dir):
            if prefix in file_name:
                # load list of results
                with open(f'{args.rcd_dir}/{file_name}', 'rb') as f:
                    tmp = pickle.load(f)    
                for line in tmp:    # each line : [dataset_name, modality, sample_id, scores_of_labels(dict), label_names] 
                    evaluated_samples.add(f'{line[0]}_{line[2]}')
                    
    # dataset and loader - use MPL-specific classes
    # MPL uses 96x96x96 patches instead of SAT's crop_size
    mpl_patch_size = [96, 96, 96] 
    norm_perc = getattr(args, 'norm_perc', 100)  # Default normalization percentile
    
    if is_master():
        print(f'** MPL Evaluation ** : Using patch size {mpl_patch_size}')
        print(f'** MPL Evaluation ** : Normalization percentile {norm_perc}')
    
    if args.online_crop:
        testset = Evaluate_Dataset_OnlineCrop_MPL(
            args.datasets_jsonl, 
            args.max_queries, 
            args.batchsize_3d, 
            mpl_patch_size,
            norm_perc,
            evaluated_samples
        )
    else:
        testset = Evaluate_Dataset_MPL(
            args.datasets_jsonl, 
            args.max_queries, 
            args.batchsize_3d, 
            mpl_patch_size,
            norm_perc, 
            evaluated_samples
        )
    sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, sampler=sampler, batch_size=1, pin_memory=args.pin_memory, num_workers=args.num_workers, collate_fn=collate_fn_mpl, shuffle=False)
    sampler.set_epoch(0)
    
    # set model for MPL integration
    model = build_maskformer_MPLseg(args, device, gpu_id)
    
    # load knowledge encoder
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        partial_load=args.text_encoder_partial_load,
        open_bert_layer=12,
        open_modality_embed=False,
        gpu_id=gpu_id,
        device=device
    )
    
    # load checkpoint using MPL-specific loader
    if args.checkpoint:
        if is_master():
            print(f'** MPL Evaluation ** : Loading checkpoint from {args.checkpoint}')
        
        # Use MPL-specific checkpoint loading
        unet_checkpoint = getattr(args, 'unet_checkpoint', None)
        model, _, _ = load_checkpoint_MPLseg(
            args.checkpoint,  # MPL checkpoint
            unet_checkpoint,  # UNET checkpoint for transformer decoder
            args.resume, 
            args.partial_load,
            model,
            device
        )
        
        if is_master():
            print('** MPL Evaluation ** : Checkpoint loaded successfully')
    else:
        if is_master():
            print('** Warning ** : No checkpoint specified, using random weights')
    
    # choose how to evaluate the checkpoint - use MPL-specific evaluator
    evaluate_mpl(model=model,
                 text_encoder=text_encoder,
                 device=device,
                 testset=testset,
                 testloader=testloader,
                 csv_path=csv_path,
                 resume=args.resume,
                 save_interval=args.save_interval,
                 dice_score=args.dice,
                 nsd_score=args.nsd,
                 visualization=args.visualization)

if __name__ == '__main__':
    # get configs
    args = parse_args_mapseg()
    
    # Override default settings for MPL evaluation if needed
    if is_master():
        print('** MPL Evaluation ** : Starting evaluation with MPL-integrated SAT model')
        if hasattr(args, 'vision_backbone'):
            print(f'** MPL Evaluation ** : Vision backbone: {args.vision_backbone}')
    
    main(args)