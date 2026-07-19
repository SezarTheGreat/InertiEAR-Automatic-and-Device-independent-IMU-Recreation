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

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        # Enable mixed precision training (bfloat16 for CPU, float16 for CUDA)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cpu" else torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cpu" else torch.float16):
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
    parser = argparse.ArgumentParser(description="Train InertiEAR DenseNet Classifier")
    parser.add_argument("--csv_file", type=str, default="StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--data_dir", type=str, default="StealthyIMU_dataset",
                        help="Data directory")
    parser.add_argument("--cache_dir", type=str, default="StealthyIMU_dataset/processed_cache",
                        help="Cache directory for features")
    parser.add_argument("--epochs", type=str, default="5",
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=str, default="64",
                        help="Batch size")
    parser.add_argument("--lr", type=str, default="0.01",
                        help="Initial learning rate")
    parser.add_argument("--seed", type=str, default="42",
                        help="Random seed")
    parser.add_argument("--growth_rate", type=str, default="12",
                        help="DenseNet growth rate")
    parser.add_argument("--use_segmentation", action="store_true", default=True,
                        help="Use coherence-based segmentation")
    parser.add_argument("--num_workers", type=str, default="4",
                        help="Number of dataloader workers")
    parser.add_argument("--preload_ram", action="store_true", default=False,
                        help="Load cached spectrogram features into RAM")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint.pth to resume training from")
    
    args = parser.parse_args()
    
    # Configure CPU threads for 100% core and thread utilization
    num_cpus = multiprocessing.cpu_count()
    torch.set_num_threads(num_cpus)
    print(f"Configured PyTorch to use {num_cpus} CPU threads (all available cores).")
    
    # Cast arguments
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    seed = int(args.seed)
    growth_rate = int(args.growth_rate)
    num_workers = int(args.num_workers)
    
    set_seed(seed)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize Model
    # We use smaller block_config (e.g. 3, 6, 12, 8) for faster CPU training
    # Standard DenseNet-121 block_config is (6, 12, 24, 16)
    block_config = (3, 6, 12, 8)
    model = InertiEAR_DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        dropout=0.3,
        num_classes=7
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_piecewise_momentum_optimizer(model, base_lr=lr)
    # Milestones scheduled at epoch 30, 60, 80 as in typical papers, scaled based on milestones parameter
    scheduler = get_lr_scheduler(optimizer, milestones=[30, 60, 80], gamma=0.1)
    
    best_val_acc = 0.0
    start_epoch = 1
    
    checkpoint_path = "checkpoint.pth"
    best_model_path = "best_model.pth"
    
    # Load resume checkpoint if specified
    resume_file = args.resume
    if resume_file and os.path.exists(resume_file):
        print(f"Resuming training from checkpoint: {resume_file}")
        checkpoint = torch.load(resume_file, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"Successfully loaded checkpoint dict. Resuming at Epoch {start_epoch}")
        else:
            # Fallback for raw state dictionary files
            model.load_state_dict(checkpoint)
            start_epoch = 4 # completed epoch 3 on previous run
            best_val_acc = 0.4096
            print(f"Successfully loaded raw model state dict. Resuming at Epoch {start_epoch}")
    else:
        print(f"Initialized DenseNet with growth_rate={growth_rate}, block_config={block_config}")
        
    print("\nStarting Training...")
    total_training_start = time.time()
    epoch_times = []
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_top1, val_top3, val_top5, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        
        # Calculate ETAs
        avg_epoch_time = np.mean(epoch_times)
        epochs_remaining = epochs - epoch
        eta_sec = avg_epoch_time * epochs_remaining
        
        # Format duration and ETA
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc
        }, checkpoint_path)
        
        # Save best model
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(model.state_dict(), best_model_path)
            print(f" => Saved new best model checkpoint to {best_model_path}")
            
    total_training_duration = time.time() - total_training_start
    print(f"\nTraining completed in {format_time(total_training_duration)}")
    
    # Load best model for evaluation on test set
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final evaluation...")
        model.load_state_dict(torch.load(best_model_path))
        
    print("\nEvaluating on Test Set...")
    test_loss, test_top1, test_top3, test_top5, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc (Top1/3/5): {test_top1*100:.2f}%/{test_top3*100:.2f}%/{test_top5*100:.2f}%")
    
    # Class map labels
    target_names = [k for k, v in sorted(CLASS_MAP.items(), key=lambda item: item[1])]
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, labels=list(range(len(target_names))), target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()
