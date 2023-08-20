import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy

import h5py
import os

from collections import defaultdict
from utils.data.data_augment import DataAugmentor
from unet_utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from unet_utils.model.ResUnet import ResUnet
from med_seg_diff_pytorch import Unet, MedSegDiff

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    count = 0
    for iter, data in enumerate(data_loader):
#         if iter > 1:
#             break
        input_, target, maximum, fname, slices = data
        # [ADD] by yxxshin (2023.07.22)
        brain_mask_h5 = h5py.File(os.path.join('/root/brain_mask/train', fname[0]), 'r')
        brain_mask = torch.from_numpy(brain_mask_h5['image_mask'][()])[slices[0]]
        brain_mask = brain_mask.cuda(non_blocking=True)

        if args.is_grappa != 'y' and args.given_grappa != 'y':
            input_ = input_.unsqueeze(0)
        input_ = input_.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        
    
        if loss_type(input_[:,0]*brain_mask, target*brain_mask, maximum).item() < args.threshold:
            continue
        
        loss = model(target * brain_mask, input_ * brain_mask)

        loss.backward()
        total_loss += loss.item()
        
        if count % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
        count = count + 1
    total_loss = total_loss / count

    #wandb.log({"Train_Loss": total_loss})
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader, loss_type):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
#             if iter > 0:
#                 break
            if iter % 20 != 0:
                continue
            input_, target, maximum, fnames, slices = data
            if args.is_grappa != 'y' and args.given_grappa != 'y':
                input_ = input_.unsqueeze(0)
            input_ = input_.cuda(non_blocking=True)
            
            output = model.sample(input_).squeeze(0)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)

            for i in range(1):
            #only batch1 case?
                 # [ADD] by yxxshin (2023.07.30)
                brain_mask_h5 = h5py.File(os.path.join('/root/brain_mask/val', fnames[i]), 'r')
                brain_mask = torch.from_numpy(brain_mask_h5['image_mask'][()])
                brain_mask = brain_mask.cuda(non_blocking=True)
                
                output[i] = output[i] * brain_mask[slices[0]]
                target[i] = target[i] * brain_mask[slices[0]]

                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                # targets[fnames[i]][int(slices[i])] = target[i].numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)



def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

#     model = ResUnet(args.chanels)
#     model.to(device=device)
    unet = Unet(
    dim = 32,
    image_size = 384,
    mask_channels = 1,          # segmentation has 1 channel
    input_img_channels = args.chanels,     # input images have 3 channels
    dim_mults = (1, 2, 4, 8)
)
    model = MedSegDiff(unet,timesteps = 100).cuda()

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, mode='train', shuffle=True)
    #train_loader = create_data_loaders(data_path = args.data_path_train, args = args, mode='train')
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, mode='val', shuffle=True)
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, loss_type)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)


        val_loss = val_loss / num_subjects
        #wandb.log({"Valid_Loss": val_loss})

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )