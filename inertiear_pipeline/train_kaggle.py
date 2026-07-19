import os
import argparse
import random
import multiprocessing
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

from inertiear_pipeline.dataset import InertiEARDataset, CLASS_MAP
from inertiear_pipeline.model import InertiEAR_DenseNet, get_piecewise_momentum_optimizer, get_lr_scheduler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero gradients with set_to_none=True to conserve memory
        optimizer.zero_grad(set_to_none=True)
        
        # AMP Autocast (FP16 on GPU, BF16 on CPU)
        device_type = device.type if hasattr(device, 'type') else 'cuda'
        autocast_dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16
        
        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        if scaler is not None and device_type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    
    all_outputs = []
    all_labels = []
    
    device_type = device.type if hasattr(device, 'type') else 'cuda'
    autocast_dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        
        all_outputs.append(outputs.cpu())
        all_labels.extend(labels.cpu().numpy())
        
    val_loss = running_loss / total
    
    # Concatenate all outputs
    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    # Calculate top-k accuracy
    top1, top3, top5 = 0.0, 0.0, 0.0
    if total > 0:
        maxk = 5
        _, pred = all_outputs_tensor.topk(maxk, 1, True, True)
        pred = pred.t() # shape (5, total)
        correct = pred.eq(all_labels_tensor.view(1, -1).expand_as(pred))
        
        correct_1 = correct[:1].reshape(-1).float().sum(0).item()
        correct_3 = correct[:3].reshape(-1).float().sum(0).item()
        correct_5 = correct[:5].reshape(-1).float().sum(0).item()
        
        top1 = correct_1 / total
        top3 = correct_3 / total
        top5 = correct_5 / total
        
    # Predictions (Argmax)
    all_preds = list(all_outputs_tensor.argmax(dim=1).numpy())
    
    return val_loss, top1, top3, top5, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Train InertiEAR DenseNet on Kaggle (Optimized for Multi-GPU)")
    parser.add_argument("--csv_file", type=str, default="/kaggle/input/datasets/sezarthegreat/stealthyimu-dataset/StealthyIMU_dataset/metadata/stealthyIMU_all.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/datasets/sezarthegreat/stealthyimu-dataset/StealthyIMU_dataset",
                        help="Data directory")
    parser.add_argument("--cache_dir", type=str, default="/kaggle/working/processed_cache",
                        help="Cache directory for features")
    parser.add_argument("--epochs", type=str, default="50",
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=str, default="256",
                        help="Batch size (large batch to saturate Multi-GPU)")
    parser.add_argument("--lr", type=str, default="0.01",
                        help="Initial learning rate")
    parser.add_argument("--seed", type=str, default="42",
                        help="Random seed")
    parser.add_argument("--growth_rate", type=str, default="12",
                        help="DenseNet growth rate")
    parser.add_argument("--use_segmentation", action="store_true", default=True,
                        help="Use coherence-based segmentation")
    parser.add_argument("--num_workers", type=str, default="2",
                        help="Optimal loader workers for 4 vCPUs")
    parser.add_argument("--preload_ram", action="store_true", default=False,
                        help="Whether to preload dataset in RAM (disable to prevent OOM)")
    parser.add_argument("--resume", type=str, default="best_model.pth",
                        help="Path to checkpoint file to resume training from")
    
    args = parser.parse_args()
    
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    seed = int(args.seed)
    growth_rate = int(args.growth_rate)
    num_workers = int(args.num_workers)
    
    set_seed(seed)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    # Load dataset
    print(f"Loading dataset from: {args.csv_file}")
    full_dataset = InertiEARDataset(
        csv_file=args.csv_file,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        use_segmentation=args.use_segmentation,
        preload_ram=args.preload_ram
    )
    
    # Split: 80% train, 10% validation, 10% test
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset split size -> Train: {train_len}, Val: {val_len}, Test: {test_len}")
    
    # DataLoaders optimized with pin_memory=True
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize Model with memory_efficient=True (gradient checkpointing)
    block_config = (3, 6, 12, 8)
    model = InertiEAR_DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        dropout=0.3,
        num_classes=7,
        memory_efficient=True
    )
    
    # Move model and wrap with DataParallel if multiple GPUs
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Wrapping model in nn.DataParallel across {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_piecewise_momentum_optimizer(model, base_lr=lr)
    scheduler = get_lr_scheduler(optimizer, milestones=[30, 60, 80], gamma=0.1)
    
    # PyTorch GradScaler for FP16 training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_val_acc = 0.0
    start_epoch = 1
    
    checkpoint_path = "checkpoint.pth"
    best_model_path = "best_model.pth"
    
    # Load resume checkpoint if specified
    resume_file = args.resume
    if resume_file and os.path.exists(resume_file):
        print(f"Resuming training from checkpoint: {resume_file}")
        checkpoint = torch.load(resume_file, map_location=device)
        
        # Load state dictionary
        state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
        cleaned_sd = clean_state_dict(state_dict)
        
        if isinstance(model, nn.DataParallel):
            dp_sd = {f"module.{k}": v for k, v in cleaned_sd.items()}
            model.load_state_dict(dp_sd)
        else:
            model.load_state_dict(cleaned_sd)
            
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"Successfully loaded checkpoint dict. Resuming at Epoch {start_epoch}")
        else:
            start_epoch = 4
            best_val_acc = 0.4096
            print(f"Successfully loaded raw model state dict. Resuming at Epoch {start_epoch}")
    else:
        print(f"Initialized DenseNet with growth_rate={growth_rate}, block_config={block_config}, memory_efficient=True")
        
    print("\nStarting Training...")
    total_training_start = time.time()
    epoch_times = []
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_top1, val_top3, val_top5, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        
        avg_epoch_time = np.mean(epoch_times)
        epochs_remaining = epochs - epoch
        eta_sec = avg_epoch_time * epochs_remaining
        
        def format_time(seconds):
            mins, secs = divmod(int(seconds), 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                return f"{hours:02d}h:{mins:02d}m:{secs:02d}s"
            return f"{mins:02d}m:{secs:02d}s"
            
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc (Top1/3/5): {val_top1*100:.1f}%/{val_top3*100:.1f}%/{val_top5*100:.1f}% | "
              f"Time: {format_time(epoch_duration)} | ETA: {format_time(eta_sec)}")
              
        # Save general checkpoint
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc
        }, checkpoint_path)
        
        # Save best model
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f" => Saved new best model checkpoint to {best_model_path}")
            
    total_training_duration = time.time() - total_training_start
    print(f"\nTraining completed in {format_time(total_training_duration)}")
    
    # Load best model for evaluation on test set
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final evaluation...")
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(torch.load(best_model_path))
        
    print("\nEvaluating on Test Set...")
    test_loss, test_top1, test_top3, test_top5, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc (Top1/3/5): {test_top1*100:.2f}%/{test_top3*100:.2f}%/{test_top5*100:.2f}%")
    
    target_names = [k for k, v in sorted(CLASS_MAP.items(), key=lambda item: item[1])]
    print("\nClassification Report:")
    print(classification_report(labels, preds, labels=list(range(len(target_names))), target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()
