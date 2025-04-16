import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import math
import time

from config import TrainingConfig
from models.clip_model import ClipVisionModel, load_clip_model
# from attacks.pgd import pgd
from attacks.pgd_huber import pgd
from utils.logger import logger


def train_one_epoch(step_total, model, model_orig, dataloader, optimizer, scheduler, config):
    model.train()
    model_orig.eval()
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.cuda(), targets.cuda()
        
        with torch.no_grad():
            embedding_orig = model_orig(data, output_normalize=config.output_normalize)
        
        
        data_adv = pgd(
            forward=model,
            loss_fn=lambda x, _: F.mse_loss(x, embedding_orig, reduction='mean'),
            data_clean=data,
            targets=targets,
            norm=config.norm,
            eps=config.eps,
            iterations=config.iterations_adv,
            stepsize=config.stepsize_adv,
            output_normalize=config.output_normalize
        )
        
        embedding = model(data_adv, output_normalize=config.output_normalize)
        loss_adv = F.mse_loss(embedding, embedding_orig, reduction='none').mean(dim=1).mean()
        
        optimizer.zero_grad()
        loss_adv.backward()
        optimizer.step()
        scheduler.step()
        
        
        if batch_idx % config.log_freq == 0:
            logger.info(f'Train Step: {step_total}/{config.steps} '
                  f'({100. * step_total / config.steps:.0f}%)\tLoss: {loss_adv.item():.6f}')
        
        step_total += 1
        if step_total >= config.steps:
            break
    
    return step_total

def train(config: TrainingConfig):
    logger.info("Initializing training...")


    model_orig, preprocessor, normalizer = load_clip_model(config.clip_model_name)
    model, _, _ = load_clip_model(config.clip_model_name)

    logger.info(f"Loading dataset from: {config.imagenet_root}")
    dataset = ImageFolder(
        root=os.path.join(config.imagenet_root, 'train'),
        transform=preprocessor
    )
    logger.info(f"Training samples: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)
    logger.info(f"Created dataloader with batch size: {config.batch_size}")

    # Extracting Vision Encoder Only
    logger.info("Setting up models...")
    model_orig = ClipVisionModel(model=model_orig.visual, normalize=normalizer)
    model = ClipVisionModel(model=model.visual, normalize=normalizer)
    
    logger.info("Models moved to CUDA")
    model_orig.cuda()
    model.cuda()
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        model_orig = nn.DataParallel(model_orig)
    else:
        logger.info("Using single GPU")
    
    
    logger.info(f"Setting up {config.opt} optimizer...")
    params = model.parameters()
    optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.wd)
    logger.info(f"Optimizer: {optimizer}")

    logger.info("Setting up cosine learning rate scheduler...")
    def cosine_lr(optimizer, base_lr, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = cosine_lr(optimizer, config.lr, config.warmup, config.steps)
    logger.info(f"Learning rate scheduler: cosine with warmup={config.warmup}, total_steps={config.steps}")

    logger.info("Starting training loop...")
    step_total = 0
    start_time = time.time()
    
    while step_total < config.steps:
        step_total = train_one_epoch(
            step_total,
            model=model,
            model_orig=model_orig,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config
        )

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    logger.info(f"Created output directory: {config.output_dir}")

    if config.save_checkpoints:
        if isinstance(model, nn.DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
            
        checkpoint_path = os.path.join(config.output_dir, 'checkpoints/final2.pt')
        optimizer_path = os.path.join(config.output_dir, 'checkpoints/final_opt2.pt')
        torch.save(model_to_save.state_dict(), checkpoint_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        logger.info(f"Saved final model checkpoint to {checkpoint_path}")
        logger.info(f"Saved optimizer state to {optimizer_path}")
