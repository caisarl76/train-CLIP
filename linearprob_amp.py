import argparse
import builtins

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
import torch.distributed as dist
import torch.nn as nn

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler 
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM, CLIPConfig
import albumentations as A
from pycocotools.coco import COCO
from PIL import Image

from data.custom_data import *

data_root = '/data/aihub/Training/images'  
label_root = '/data/aihub/Training/labels/labels.json'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/data/aihub/Training/images', type=str )
parser.add_argument('--label_root', default='/data/aihub/Training/labels/labels.json', type=str)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=1e-05, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--workers', default=0, type=int, )
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--multiprocessing-distributed', default=True, type=bool,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def main():
    args = parser.parse_args()
    if not args.dist_url:
        args.dist_url = 'tcp://localhost:' + (str)(np.random.randint(9000, 11000, 1)[0])
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(run_training, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    
def run_training(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
    # Load the CLIP model and processor
    # model_name = "openai/clip-vit-base-patch32"
    model_name = "EleutherAI/polyglot-ko-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    for name, param in model.named_parameters():
        if "visual" in name and "projection" not in name:
            param.requires_grad = False
        elif "text" in name and "projection" not in name:
            param.requires_grad = False
    
    optimizer = AdamW(
        [
            {'params':model.visual_projection.parameters()},
            {'params':model.text_projection.parameters()}
            ],
        lr=0.0001
        )
    criterion = nn.CosineEmbeddingLoss()

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node -1) / ngpus_per_node)
            device = torch.device("cuda", args.gpu)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            device = torch.device("cuda", args.gpu)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        device = torch.device("cuda", args.gpu)
    else:
        raise NotImplementedError("Only DDP supported")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
    
    train_dataset = AIHub_data(root=data_root, json=label_root, transform=get_transforms(mode='train'), tokenizer=tokenizer)
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
    
    scaler = GradScaler()

    # Training loop
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        sampler.set_epoch(epoch)
        for batch, inputs in enumerate(train_loader):
            inputs = {k:v.to('cuda:1') for k, v in inputs.items()}
            optimizer.zero_grad()
            # Preprocess images
            with autocast():
                outputs = model(**inputs)
                
                batch_size = inputs['pixel_values'].size(0)
                logits_per_image = outputs.logits_per_image

                # Generate random labels for contrastive learning
                targets = torch.arange(batch_size).to(device)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits_per_image / 0.1, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            
            if (batch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{10}], Step [{batch+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{10}], Total Loss: {total_loss:.4f}")
        # break

    # Save the trained model
    if not os.path.exists('./runs/'):
        os.makedirs('./runs')
    model.module.save_pretrained("./runs/final_model.ckpt")

if __name__=="__main__":
    main()