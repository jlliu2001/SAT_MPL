import os
import time
import numpy as np

import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from .dist import is_master
from .loss import segmentation_loss
import nibabel as nib


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1): # avg value over samples, and the num of samples, e.g. the avg loss over a batch and batchsize
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class Trainer():
    def __init__(self,
                 args, 
                 model,
                 text_encoder,
                 device, 
                 trainset, 
                 trainloader, 
                 sampler, 
                 dice_loss, 
                 bce_w_logits_loss, 
                 optimizer, 
                 scheduler, 
                 tb_writer, 
                 checkpoint_dir, 
                 log_file):
        
        # average meters
        self.dataset_dice_loss_m_dict = {}
        self.dataset_ce_loss_m_dict = {}
        self.dice_loss_m = AverageMeter()
        self.ce_loss_m = AverageMeter()
        self.data_time_m = AverageMeter()
        self.query_time_m = AverageMeter()
        self.compute_time_m = AverageMeter()
        self.loss_time_m = AverageMeter()
        self.bp_time_m = AverageMeter()
        self.sample_statistics = {}

        # model
        self.model = model
        self.text_encoder = text_encoder
        self.device = device
        
        # data
        self.trainset = trainset
        self.trainloader = iter(trainloader)    
        self.sampler = sampler 
        
        # loss calculator, optimizer and lr
        self.dice_loss = dice_loss 
        self.bce_w_logits_loss = bce_w_logits_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # log
        self.tb_writer = tb_writer
        self.log_file = log_file
        self.total_steps = args.step_num
        if isinstance(self.total_steps, list):  # could be multi-stage restart lr
            self.total_steps = sum(self.total_steps)
        self.log_step_interval = args.log_step_interval
        if self.log_step_interval is None:
            self.log_step_interval = 1000
        
        # checkpoint   
        self.checkpoint_dir = checkpoint_dir
        self.save_large_interval = args.save_large_interval
        self.save_small_interval = args.save_small_interval
            
        # amp
        self.grad_scaler = GradScaler()
        
        # accumulate grad
        self.accumulation_steps = 0
        self.accumulate_grad_interval = args.accumulate_grad_interval
        
        self.model.train()
        
    def train_one_step(self, step):
        
        end_time = time.time()        

        # step-wise scheduling
        self.scheduler(step)
            
        # data loading
        batch = next(self.trainloader)
        text, modality, image, mask, dataset_name, query_mask = batch['text'], batch['modality'], batch['image'], batch['mask'], batch['dataset'], batch['query_mask']
        batch_size = len(text)
        image = image.to(device=self.device)
        
        mask = mask.to(device=self.device)
        query_mask = query_mask.to(device=self.device)   # B C
        
        if is_master():
            self.data_time_m.update(time.time()-end_time)
            end_time = time.time()
        
        with autocast():
            # get query embeddings from label name and image modality
            queries = self.text_encoder(text, modality)
            
            if is_master():
                self.query_time_m.update(time.time()-end_time)
                end_time = time.time()
        
            # forward
            logits = self.model(queries=queries, image_input=image)   # bnhwd or list of bnhwd (deep supervision)
            
            if is_master():
                self.compute_time_m.update(time.time()-end_time)
                end_time = time.time()
            
            # generate weight for each layer output
            weights = torch.Tensor([1 / (2 ** i) for i in range(len(logits))])  # 1, 1/2, 1/4, 1/8
            weights = weights / weights.sum()
            # loss of the largest output
            bp_loss, unreduced_batch_dice_loss, unreduced_batch_ce_loss = segmentation_loss(logits[0], mask, query_mask, self.dice_loss, self.bce_w_logits_loss, weights[0])
            # loss of mid-stage output
            for mid_logits, mid_weight in zip(logits[1:], weights[1:]):
                mid_mask = torch.nn.functional.interpolate(mask, size=mid_logits.shape[2:], mode='nearest')  # HWD -> hwd
                tmp_bp_loss, tmp_unreduced_batch_dice_loss, tmp_unreduced_batch_ce_loss = segmentation_loss(mid_logits, mid_mask, query_mask, self.dice_loss, self.bce_w_logits_loss, mid_weight)
                bp_loss += tmp_bp_loss
                unreduced_batch_dice_loss += tmp_unreduced_batch_dice_loss
                unreduced_batch_ce_loss += tmp_unreduced_batch_ce_loss
        
        # scale the loss
        self.grad_scaler.scale(bp_loss).backward()
        self.accumulation_steps += 1
        
        if is_master():
            self.loss_time_m.update(time.time()-end_time)
            end_time = time.time()
        
        # accumulate grad or bp and update the model
        if self.accumulation_steps % self.accumulate_grad_interval == 0:
            self.grad_scaler.step(self.optimizer)
            self.optimizer.zero_grad()
            self.grad_scaler.update()
            self.accumulation_steps = 0
        
        if is_master():
            self.bp_time_m.update(time.time()-end_time)
            end_time = time.time()

        with torch.no_grad():
            if is_master():
                # attribute loss to each dataset
                for i, name in enumerate(dataset_name):
                    if name not in self.dataset_dice_loss_m_dict:
                        self.dataset_dice_loss_m_dict[name] = AverageMeter()
                        self.dataset_ce_loss_m_dict[name] = AverageMeter()
                    self.dataset_dice_loss_m_dict[name].update(unreduced_batch_dice_loss[i].item(), 1)
                    self.dataset_ce_loss_m_dict[name].update(unreduced_batch_ce_loss[i].item(), 1)
                    if name not in self.sample_statistics:
                        self.sample_statistics[name] = 0
                    self.sample_statistics[name] += 1
                # overall loss
                self.dice_loss_m.update(torch.mean(unreduced_batch_dice_loss).item(), batch_size)
                self.ce_loss_m.update(torch.mean(unreduced_batch_ce_loss).item(), batch_size)
            
                # log and save regularly (and only after a update of the model in case of accmulation grad)
                if self.accumulation_steps == 0:
                    
                    # log regularly
                    if step % self.log_step_interval == 0:  
                        
                        # tensorboard
                        lr = self.optimizer.param_groups[0]['lr']
                        self.tb_writer.add_scalar('train_dice_loss/all_dataset', self.dice_loss_m.avg, step)
                        self.tb_writer.add_scalar('train_ce_loss/all_dataset', self.ce_loss_m.avg, step)
                        self.tb_writer.add_scalar('train/learning_rate', lr, step)
                        for name, meter in self.dataset_dice_loss_m_dict.items():
                            self.tb_writer.add_scalar(f'train_dice_loss/{name}', meter.avg, step)
                        for name, meter in self.dataset_ce_loss_m_dict.items():
                            self.tb_writer.add_scalar(f'train_ce_loss/{name}', meter.avg, step)
                        
                        # print log    
                        info = f"\nStep {step}({(step/self.total_steps):.3f}) | LR {lr:.4e} "
                        info += f"| Dice Loss {self.dice_loss_m.val:.4f}({self.dice_loss_m.avg:.4f}) | CE Loss {self.ce_loss_m.val:.4f}({self.ce_loss_m.avg:.4f}) |"
                        info += f"| Data Time {self.data_time_m.avg:.2f} | Generate Query Time {self.query_time_m.avg:.2f} | FW Time {self.compute_time_m.avg:.2f} |"
                        info += f"| Loss Time {self.loss_time_m.avg:.2f} | BP Time {self.bp_time_m.avg:.2f}\n"
                        print(info)
                        
                        # write log
                        with open(self.log_file, 'a') as f:
                            f.write(info)
                            # dice loss
                            f.write(f'| Dice Loss |')
                            sorted_keys = sorted(self.dataset_dice_loss_m_dict.keys())
                            for name in sorted_keys:
                                meter = self.dataset_dice_loss_m_dict[name]
                                f.write(f' {name} {meter.avg} |')
                            # bce loss
                            f.write(f'\n| BCE Loss |')
                            sorted_keys = sorted(self.dataset_ce_loss_m_dict.keys())
                            for name in sorted_keys:
                                meter = self.dataset_ce_loss_m_dict[name]
                                f.write(f' {name} {meter.avg} |')
                            f.write(f'\n')
                            # sample statistics
                            f.write(f'\n| Sample Statistics |')
                            sorted_keys = sorted(self.sample_statistics.keys())
                            for name in sorted_keys:
                                sampled_times = self.sample_statistics[name]
                                size, repeated_times = self.trainset.get_size_and_repeat(name)
                                f.write(f' {name} {sampled_times}/{size}/{size+repeated_times} |')
                            f.write(f'\n')
                            
                        # reset records
                        for name, meter in self.dataset_dice_loss_m_dict.items():
                            meter.reset()
                        for name, meter in self.dataset_ce_loss_m_dict.items():
                            meter.reset()
                        for k in self.sample_statistics.keys():
                            self.sample_statistics[k] = 0
                        self.dice_loss_m.reset()
                        self.ce_loss_m.reset()
                        self.data_time_m.reset()
                        self.query_time_m.reset()
                        self.compute_time_m.reset()
                        self.loss_time_m.reset()
                        self.bp_time_m.reset()
                   
                    # save regularly          
                    if step % self.save_large_interval == 0:
                        t0 = time.time()
                        torch.save({'step':step,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),         
                                    }, os.path.join(self.checkpoint_dir, f'step_{step}.pth'))
                        tmp_path = os.path.join(self.checkpoint_dir, f'text_encoder_step_{step}.pth')
                        torch.save({'step':step,
                                    'model_state_dict': self.text_encoder.model.state_dict(),
                                    }, tmp_path)
                        print(f'Save, time {time.time()-t0}s')
                    
                    # save more frequently to avoid interruption 
                    if self.save_small_interval and step % self.save_small_interval == 0:
                        t0 = time.time()
                        torch.save({'step': step,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),         
                                    }, os.path.join(self.checkpoint_dir, f'latest_step.pth'))
                        tmp_path = os.path.join(self.checkpoint_dir, f'text_encoder_latest_step.pth')
                        torch.save({'step':step,
                                    'model_state_dict': self.text_encoder.model.state_dict(),
                                    }, tmp_path)
                        print(f'Save, time {time.time()-t0}s')
                


class Trainer_MPL():
    def __init__(self,
                 args, 
                 model,
                 text_encoder,
                 device, 
                 trainset, 
                 trainloader, 
                 sampler, 
                 dice_loss, 
                 bce_w_logits_loss, 
                 optimizer, 
                 scheduler, 
                 tb_writer, 
                 checkpoint_dir, 
                 log_file):
        
        # average meters
        self.dataset_dice_loss_m_dict = {}
        self.dataset_ce_loss_m_dict = {}
        self.dice_loss_m = AverageMeter()
        self.ce_loss_m = AverageMeter()
        self.data_time_m = AverageMeter()
        self.query_time_m = AverageMeter()
        self.compute_time_m = AverageMeter()
        self.loss_time_m = AverageMeter()
        self.bp_time_m = AverageMeter()
        self.sample_statistics = {}

        # model
        self.model = model
        self.text_encoder = text_encoder
        self.device = device
        
        # data
        self.trainset = trainset
        self.trainloader = iter(trainloader) 
        # self.trainloader = trainloader
        self.sampler = sampler 
        
        # loss calculator, optimizer and lr
        self.dice_loss = dice_loss 
        self.bce_w_logits_loss = bce_w_logits_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # log
        self.tb_writer = tb_writer
        self.log_file = log_file
        self.total_steps = args.step_num
        if isinstance(self.total_steps, list):  # could be multi-stage restart lr
            self.total_steps = sum(self.total_steps)
        self.log_step_interval = args.log_step_interval
        if self.log_step_interval is None:
            self.log_step_interval = 1000
        
        # checkpoint   
        self.checkpoint_dir = checkpoint_dir
        self.save_large_interval = args.save_large_interval
        self.save_small_interval = args.save_small_interval
            
        # amp
        self.grad_scaler = GradScaler()
        
        # accumulate grad
        self.accumulation_steps = 0
        self.accumulate_grad_interval = args.accumulate_grad_interval

        img_info=nib.load("/data0/user/jlliu/git_pull_repos/CANDI13_2013-04-02_173242/1131/1131/1131/1131_1_RAS.nii.gz")
        self.img_info={}
        self.img_info['affine']=img_info.affine
        self.img_info['header']=img_info.header
        self.img_info['header']['dim']=[  3, 96, 96, 96,   1,   1,   1,   1]
        
        self.model.train()
        
    def train_one_step(self, step, samples_per_epoch=100):
        
        end_time = time.time()        

        # step-wise scheduling
        self.scheduler(step)

        epoch_dice_losses = []
        epoch_ce_losses = []
        epoch_samples_processed = 0
        
        for num_ in range(samples_per_epoch):
            try:
                # Data loading
                batch = next(self.trainloader)
                text, modality, image, local_patch, coordinates, mask, dataset_name, query_mask = batch['text'], batch['modality'], batch['image'], batch['local_patch'], batch['coordinates'], batch['mask'], batch['dataset'], batch['query_mask']
                batch_size = len(text)
                img_cpu=local_patch.detach().cpu().numpy()
                img_cpu=img_cpu[0,0]
                # np.save("/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/img_3.npy",img_cpu)
                # nib.save(nib.Nifti1Image(img_cpu,self.img_info['affine'],self.img_info['header']),"/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/img_3.nii.gz")
                nib.save(nib.nifti2.Nifti1Image(img_cpu, np.eye(4)),f"/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/img_{num_}_3.nii.gz")
                mask_cpu_=mask[0].detach().cpu().numpy()
                mask_cpu=np.zeros_like(img_cpu)
                for i in range(mask_cpu_.shape[0]):
                    tmp_=mask_cpu_[i]
                    mask_cpu[tmp_>0]=i+1

                # np.save("/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/gt_3.npy",mask_cpu)
                # nib.save(nib.Nifti1Image(mask_cpu,self.img_info['affine'],self.img_info['header']),"/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/gt_3.nii.gz")
                nib.save(nib.nifti2.Nifti1Image(mask_cpu, np.eye(4)),f"/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/gt_{num_}_3.nii.gz")
                image = image.to(device=self.device)
                local_patch = local_patch.to(device=self.device)
                coordinates = coordinates.to(device=self.device)
                mask = mask.to(device=self.device)
                query_mask = query_mask.to(device=self.device)
                
                with autocast():
                    # Get query embeddings
                    queries = self.text_encoder(text, modality)
                    
                    # Forward pass
                    logits, local_mask = self.model(queries=queries, image_input=image, local_patch=local_patch, coordinates=coordinates)
                    local_mask_=local_mask[:,1:]
                    print('local_mask_.shape:',local_mask_.shape)
                    print('logits.len:',len(logits))


                    mask_radios=[]
                    for i in range(mask.shape[1]):
                        tmp_=mask[:,i]
                        count_=torch.count_nonzero(tmp_)
                        mask_radios.append(count_/(96*96*96))

                    print('mask_radios:',mask_radios)

                    # Generate weights for multi-scale outputs
                    weights = torch.Tensor([1 / (2 ** i) for i in range(len(logits))])
                    weights = weights / weights.sum()
                    print(f'+++++++++++++loss calculation for {num_}+++++++++++')
                    # Calculate loss for the largest output
                    bp_loss, unreduced_batch_dice_loss, unreduced_batch_ce_loss = segmentation_loss(
                        logits[0], mask, query_mask, self.dice_loss, self.bce_w_logits_loss, weights[0]
                    )
                    print('**************next is local mask s dice*******************')
                    bp_loss_, unreduced_batch_dice_loss_, unreduced_batch_ce_loss_ = segmentation_loss(
                        local_mask_, mask, query_mask, self.dice_loss, self.bce_w_logits_loss, weights[0]
                    )
                    print('unreduced_batch_dice_loss_:',unreduced_batch_dice_loss_)
                    # if unreduced_batch_dice_loss_[0]<0.2:
                    local_mask_cpu=local_mask.detach().cpu().numpy()
                    local_mask_cpu=local_mask_cpu[0]
                    local_mask_cpu=np.argmax(local_mask_cpu,axis=0)
                    # np.save("/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/local_mask_3.npy",local_mask_cpu)
                    # nib.save(nib.Nifti1Image(local_mask_cpu,self.img_info['affine'],self.img_info['header']),"/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/local_mask_3.nii.gz")
                    nib.save(nib.nifti2.Nifti1Image(local_mask_cpu.astype(np.float32), np.eye(4)),f"/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/local_mask_{num_}_3.nii.gz")
                    logit_cpu=torch.sigmoid(logits[0])
                    logit_cpu=logit_cpu.detach().cpu().numpy()
                    logit_cpu=logit_cpu[0]
                    logit_cpu_=np.zeros((96,96,96))
                    for i_ in range(logit_cpu.shape[0]):
                        logit_cpu_[logit_cpu[i_]>0.5]=i_+1
                    # np.save("/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/seg_3.npy",logit_cpu)
                    nib.save(nib.nifti2.Nifti1Image(logit_cpu_.astype(np.float32), np.eye(4)),f"/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test/seg_{num_}_3.nii.gz")

                    
                    # Calculate loss for mid-stage outputs
                    # for mid_logits, mid_weight in zip(logits[1:], weights[1:]):
                    #     mid_mask = torch.nn.functional.interpolate(mask, size=mid_logits.shape[2:], mode='nearest')
                    #     tmp_bp_loss, tmp_unreduced_batch_dice_loss, tmp_unreduced_batch_ce_loss = segmentation_loss(
                    #         mid_logits, mid_mask, query_mask, self.dice_loss, self.bce_w_logits_loss, mid_weight
                    #     )
                    #     bp_loss += tmp_bp_loss
                    #     unreduced_batch_dice_loss += tmp_unreduced_batch_dice_loss
                    #     unreduced_batch_ce_loss += tmp_unreduced_batch_ce_loss
                
                # Accumulate loss for averaging
                epoch_dice_losses.extend(unreduced_batch_dice_loss.detach().cpu().numpy())
                epoch_ce_losses.extend(unreduced_batch_ce_loss.detach().cpu().numpy())
                epoch_samples_processed += batch_size
                
                # Backward pass
                self.grad_scaler.scale(bp_loss).backward()
                self.accumulation_steps += 1
                
                # Update model parameters when accumulation is complete
                if self.accumulation_steps % self.accumulate_grad_interval == 0:
                    self.grad_scaler.step(self.optimizer)
                    self.optimizer.zero_grad()
                    self.grad_scaler.update()
                    self.accumulation_steps = 0
                
                # Print progress every 10 samples
                if is_master() and (i + 1) % 10 == 0:
                    current_avg_dice = sum(epoch_dice_losses) / len(epoch_dice_losses)
                    current_avg_ce = sum(epoch_ce_losses) / len(epoch_ce_losses)
                    print(f'  Sample {i+1}/{samples_per_epoch} | Avg Dice: {current_avg_dice:.4f} | Avg CE: {current_avg_ce:.4f}')
                
            except StopIteration:
                # Restart iterator if we run out of data
                print(f"Restarting data iterator at sample {i+1}")
                break
            except Exception as e:
                print(f"Error processing sample {num_+1}: {e}")
                continue
        
        # Calculate epoch average losses
        if epoch_dice_losses and epoch_ce_losses:
            avg_dice_loss = sum(epoch_dice_losses) / len(epoch_dice_losses)
            avg_ce_loss = sum(epoch_ce_losses) / len(epoch_ce_losses)
            
            if is_master():
                # Update average meters with epoch averages
                self.dice_loss_m.update(avg_dice_loss, epoch_samples_processed)
                self.ce_loss_m.update(avg_ce_loss, epoch_samples_processed)
                
                # Log epoch results  
                epoch_time = time.time() - end_time
                lr = self.optimizer.param_groups[0]['lr']
                info = f"\n=== Epoch Step {step} ({step/self.total_steps:.3f}) | LR {lr:.4e} ==="
                info += f"\n  Processed {epoch_samples_processed} samples in {epoch_time:.1f}s"
                info += f"\n  Average Dice Loss: {avg_dice_loss:.4f}"
                info += f"\n  Average CE Loss: {avg_ce_loss:.4f}"
                info += f"\n  Samples/second: {epoch_samples_processed/epoch_time:.1f}"
                print(info)
                
                # Write to log file
                if self.log_file:
                    with open(self.log_file, 'a') as f:
                        f.write(info + '\n')
                
                # Tensorboard logging
                if self.tb_writer and step % self.log_step_interval == 0:
                    self.tb_writer.add_scalar('train_dice_loss/epoch_avg', avg_dice_loss, step)
                    self.tb_writer.add_scalar('train_ce_loss/epoch_avg', avg_ce_loss, step)
                    self.tb_writer.add_scalar('train/learning_rate', lr, step)
                    self.tb_writer.add_scalar('train/samples_per_epoch', epoch_samples_processed, step)
                
                # Reset meters periodically
                if step % self.log_step_interval == 0:
                    self.dice_loss_m.reset()
                    self.ce_loss_m.reset()
                
                # Save checkpoints
                if step % self.save_large_interval == 0:
                    t0 = time.time()
                    torch.save({
                        'step': step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(self.checkpoint_dir, f'step_{step}.pth'))
                    
                    torch.save({
                        'step': step,
                        'model_state_dict': self.text_encoder.model.state_dict(),
                    }, os.path.join(self.checkpoint_dir, f'text_encoder_step_{step}.pth'))
                    print(f'Checkpoint saved in {time.time()-t0:.1f}s')
                
                if self.save_small_interval and step % self.save_small_interval == 0:
                    t0 = time.time()
                    torch.save({
                        'step': step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(self.checkpoint_dir, f'latest_step.pth'))
                    
                    torch.save({
                        'step': step,
                        'model_state_dict': self.text_encoder.model.state_dict(),
                    }, os.path.join(self.checkpoint_dir, f'text_encoder_latest_step.pth'))
                    print(f'Latest checkpoint saved in {time.time()-t0:.1f}s')
        
        else:
            if is_master():
                print(f"Warning: No valid samples processed in step {step}")
            avg_dice_loss = 0.0
            avg_ce_loss = 0.0
                    


    def train_one_step_org(self, step):
        
        end_time = time.time()        

        # step-wise scheduling
        self.scheduler(step)
        
        
            # data loading
        batch = next(self.trainloader)
        # print(batch.keys())
        text, modality, image, local_patch, coordinates, mask, dataset_name, query_mask = batch['text'], batch['modality'], batch['image'],batch['local_patch'],batch['coordinates'], batch['mask'], batch['dataset'], batch['query_mask']
        print('image_path:',batch['image_path'])
        batch_size = len(text)
        tmp_dir="/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/test"
        mask_np=mask[0].detach().cpu().numpy()
        np.save(os.path.join(tmp_dir,'gt_3.npy'),mask_np)
        image = image.to(device=self.device)
        local_patch=local_patch.to(device=self.device)
        # coordinates=coordinates.to(device=self.device)
        mask = mask.to(device=self.device)
        query_mask = query_mask.to(device=self.device)   # B C
        # print('image.shape:',image.shape)
        # print('local_patch.shape:',local_patch.shape)
        # print('coordinates.shape:',coordinates.shape)
        
        if is_master():
            self.data_time_m.update(time.time()-end_time)
            end_time = time.time()
        
        with autocast():
            # get query embeddings from label name and image modality
            queries = self.text_encoder(text, modality)
            
            if is_master():
                self.query_time_m.update(time.time()-end_time)
                end_time = time.time()
        
            # forward
            logits = self.model(queries=queries, image_input=image, local_patch=local_patch, coordinates=coordinates)   # bnhwd or list of bnhwd (deep supervision)
            
            if is_master():
                self.compute_time_m.update(time.time()-end_time)
                end_time = time.time()
            
            # generate weight for each layer output
            weights = torch.Tensor([1 / (2 ** i) for i in range(len(logits))])  # 1, 1/2, 1/4, 1/8
            weights = weights / weights.sum()
            # loss of the largest output
            bp_loss, unreduced_batch_dice_loss, unreduced_batch_ce_loss = segmentation_loss(logits[0], mask, query_mask, self.dice_loss, self.bce_w_logits_loss, weights[0])
            logits_np=logits[0].detach().cpu().numpy()
            np.save(os.path.join(tmp_dir,'seg_3.npy'),logits_np)
            # loss of mid-stage output
            # for mid_logits, mid_weight in zip(logits[1:], weights[1:]):
            #     mid_mask = torch.nn.functional.interpolate(mask, size=mid_logits.shape[2:], mode='nearest')  # HWD -> hwd
            #     tmp_bp_loss, tmp_unreduced_batch_dice_loss, tmp_unreduced_batch_ce_loss = segmentation_loss(mid_logits, mid_mask, query_mask, self.dice_loss, self.bce_w_logits_loss, mid_weight)
            #     bp_loss += tmp_bp_loss
            #     unreduced_batch_dice_loss += tmp_unreduced_batch_dice_loss
            #     unreduced_batch_ce_loss += tmp_unreduced_batch_ce_loss
        
        # scale the loss
        self.grad_scaler.scale(bp_loss).backward()
        self.accumulation_steps += 1
        
        if is_master():
            self.loss_time_m.update(time.time()-end_time)
            end_time = time.time()
        
        # accumulate grad or bp and update the model
        if self.accumulation_steps % self.accumulate_grad_interval == 0:
            self.grad_scaler.step(self.optimizer)
            self.optimizer.zero_grad()
            self.grad_scaler.update()
            self.accumulation_steps = 0
        
        if is_master():
            self.bp_time_m.update(time.time()-end_time)
            end_time = time.time()

        with torch.no_grad():
            if is_master():
                # attribute loss to each dataset
                for i, name in enumerate(dataset_name):
                    if name not in self.dataset_dice_loss_m_dict:
                        self.dataset_dice_loss_m_dict[name] = AverageMeter()
                        self.dataset_ce_loss_m_dict[name] = AverageMeter()
                    self.dataset_dice_loss_m_dict[name].update(unreduced_batch_dice_loss[i].item(), 1)
                    self.dataset_ce_loss_m_dict[name].update(unreduced_batch_ce_loss[i].item(), 1)
                    if name not in self.sample_statistics:
                        self.sample_statistics[name] = 0
                    self.sample_statistics[name] += 1
                # overall loss
                self.dice_loss_m.update(torch.mean(unreduced_batch_dice_loss).item(), batch_size)
                self.ce_loss_m.update(torch.mean(unreduced_batch_ce_loss).item(), batch_size)
            
                # log and save regularly (and only after a update of the model in case of accmulation grad)
                if self.accumulation_steps == 0:
                    
                    # log regularly
                    if step % self.log_step_interval == 0:  
                        
                        # tensorboard
                        lr = self.optimizer.param_groups[0]['lr']
                        self.tb_writer.add_scalar('train_dice_loss/all_dataset', self.dice_loss_m.avg, step)
                        self.tb_writer.add_scalar('train_ce_loss/all_dataset', self.ce_loss_m.avg, step)
                        self.tb_writer.add_scalar('train/learning_rate', lr, step)
                        for name, meter in self.dataset_dice_loss_m_dict.items():
                            self.tb_writer.add_scalar(f'train_dice_loss/{name}', meter.avg, step)
                        for name, meter in self.dataset_ce_loss_m_dict.items():
                            self.tb_writer.add_scalar(f'train_ce_loss/{name}', meter.avg, step)
                        
                        # print log    
                        info = f"\nStep {step}({(step/self.total_steps):.3f}) | LR {lr:.4e} "
                        info += f"| Dice Loss {self.dice_loss_m.val:.4f}({self.dice_loss_m.avg:.4f}) | CE Loss {self.ce_loss_m.val:.4f}({self.ce_loss_m.avg:.4f}) |"
                        info += f"| Data Time {self.data_time_m.avg:.2f} | Generate Query Time {self.query_time_m.avg:.2f} | FW Time {self.compute_time_m.avg:.2f} |"
                        info += f"| Loss Time {self.loss_time_m.avg:.2f} | BP Time {self.bp_time_m.avg:.2f}\n"
                        print(info)
                        
                        # write log
                        with open(self.log_file, 'a') as f:
                            f.write(info)
                            # dice loss
                            f.write(f'| Dice Loss |')
                            sorted_keys = sorted(self.dataset_dice_loss_m_dict.keys())
                            for name in sorted_keys:
                                meter = self.dataset_dice_loss_m_dict[name]
                                f.write(f' {name} {meter.avg} |')
                            # bce loss
                            f.write(f'\n| BCE Loss |')
                            sorted_keys = sorted(self.dataset_ce_loss_m_dict.keys())
                            for name in sorted_keys:
                                meter = self.dataset_ce_loss_m_dict[name]
                                f.write(f' {name} {meter.avg} |')
                            f.write(f'\n')
                            # sample statistics
                            f.write(f'\n| Sample Statistics |')
                            sorted_keys = sorted(self.sample_statistics.keys())
                            for name in sorted_keys:
                                sampled_times = self.sample_statistics[name]
                                size, repeated_times = self.trainset.get_size_and_repeat(name)
                                f.write(f' {name} {sampled_times}/{size}/{size+repeated_times} |')
                            f.write(f'\n')
                            
                        # reset records
                        for name, meter in self.dataset_dice_loss_m_dict.items():
                            meter.reset()
                        for name, meter in self.dataset_ce_loss_m_dict.items():
                            meter.reset()
                        for k in self.sample_statistics.keys():
                            self.sample_statistics[k] = 0
                        self.dice_loss_m.reset()
                        self.ce_loss_m.reset()
                        self.data_time_m.reset()
                        self.query_time_m.reset()
                        self.compute_time_m.reset()
                        self.loss_time_m.reset()
                        self.bp_time_m.reset()
                
                    # save regularly          
                    if step % self.save_large_interval == 0:
                        t0 = time.time()
                        torch.save({'step':step,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),         
                                    }, os.path.join(self.checkpoint_dir, f'step_{step}.pth'))
                        tmp_path = os.path.join(self.checkpoint_dir, f'text_encoder_step_{step}.pth')
                        torch.save({'step':step,
                                    'model_state_dict': self.text_encoder.model.state_dict(),
                                    }, tmp_path)
                        print(f'Save, time {time.time()-t0}s')
                    
                    # save more frequently to avoid interruption 
                    if self.save_small_interval and step % self.save_small_interval == 0:
                        t0 = time.time()
                        torch.save({'step': step,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),         
                                    }, os.path.join(self.checkpoint_dir, f'latest_step.pth'))
                        tmp_path = os.path.join(self.checkpoint_dir, f'text_encoder_latest_step.pth')
                        torch.save({'step':step,
                                    'model_state_dict': self.text_encoder.model.state_dict(),
                                    }, tmp_path)
                        print(f'Save, time {time.time()-t0}s')