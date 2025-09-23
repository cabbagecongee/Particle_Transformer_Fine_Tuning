#the following training is based on parameters specified in https://arxiv.org/pdf/2401.13536

import os
import csv
import random
import subprocess
import torch
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq
from datetime import timedelta
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from model import ParticleTransformerBackbone
from dataloader import IterableJetDataset, InterleavedJetDataset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

NAME = "multiclass_5%_ijd_plain_20e"
BATCH_SIZE = 512
LR = 1e-4
EPOCHS = 20
DATA_DIR = "/mnt/data/jet_data"
SAVE_DIR = "/mnt/data/output"

def parquet_num_rows(path: str) -> int:
    """Fast: read metadata only."""
    return pq.ParquetFile(path).metadata.num_rows

def parquet_num_rows_allowed(path: str, allowed_labels: set | None = None) -> int:
    """
    Fast-ish: scans ONLY the `jet_label` column row-group by row-group and counts
    rows in `allowed_labels`. If `allowed_labels` is None, returns total rows.
    """
    pf = pq.ParquetFile(path)
    if not allowed_labels:
        return pf.metadata.num_rows

    allowed_arr = np.array(sorted(allowed_labels), dtype=np.int64)
    total = 0
    # Iterate row groups to keep memory low
    for rg in range(pf.num_row_groups):
        col = pf.read_row_group(rg, columns=["jet_label"])["jet_label"]
        # `col` is a pyarrow Array. Convert cheaply to numpy:
        y = np.asarray(col).astype(np.int64, copy=False)
        # Count membership (vectorized)
        # Faster than np.isin for small set sizes:
        total += np.count_nonzero(np.in1d(y, allowed_arr, assume_unique=False))
    return total

filelist_path = os.path.join(DATA_DIR, "filelist.txt")
metrics_path = os.path.join(SAVE_DIR, f"training_metrics_{NAME}.csv")

kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs])


if accelerator.is_main_process:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(filelist_path):
        subprocess.run([
            "wget",
            "https://huggingface.co/datasets/jet-universe/jetclass2/resolve/main/filelist.txt",
            "-O",
            filelist_path
        ], check=True)
    
    if len(os.listdir(DATA_DIR)) <= 1:  # only filelist.txt exists
        print("Data appears empty please download data to PVC.")
        # subprocess.run(["wget", "-c", "-i", filelist_path, "-P", DATA_DIR], check=True)
# accelerator.wait_for_everyone()

with open(filelist_path, "r") as f:
    filepaths = [os.path.join(DATA_DIR, os.path.basename(line.strip())) for line in f.readlines()]

random.shuffle(filepaths)
n = len(filepaths)

train_files = filepaths[:int(0.05*n)]
val_files = filepaths[int(0.05*n):int(0.1*n)]

train_dataset = InterleavedJetDataset(train_files, batch_size=BATCH_SIZE)
val_dataset = InterleavedJetDataset(val_files, batch_size=BATCH_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

try:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size, rank = accelerator.num_processes, accelerator.process_index
except Exception:
    world_size, rank = accelerator.num_processes, accelerator.process_index

train_files_rank = train_files[rank::world_size]
val_files_rank   = val_files[rank::world_size]

length_train_rank = sum(parquet_num_rows_allowed(fp) for fp in train_files_rank)
length_val_rank   = sum(parquet_num_rows_allowed(fp) for fp in val_files_rank)

num_iterations_train = (length_train_rank + BATCH_SIZE - 1) // BATCH_SIZE
num_iterations_val   = (length_val_rank   + BATCH_SIZE - 1) // BATCH_SIZE


model = ParticleTransformerBackbone(
    input_dim=19,         
    num_classes=188,      
    use_hlfs = False,
  )

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, betas=(0.95,0.999))
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations_train * EPOCHS, eta_min=1e-7)

train_loader, val_loader, optimizer, model = accelerator.prepare(
    train_loader, val_loader, optimizer, model
)


acc = []
val_acc = []
val_losses = []
train_losses = []
best_val_loss = float('inf')
best_val_acc = 0.0
best_val_loss_epoch = -1
best_val_acc_epoch = -1

for epoch in range(EPOCHS):
  model.train()
  total, correct, total_loss = 0, 0, 0
  train_loss_acum = 0.0
  train_bar = tqdm(
        train_loader,
        total=num_iterations_train,
        desc=f"Epoch {epoch+1}/{EPOCHS} [train]",
        disable=not accelerator.is_main_process
    )
  for x_particles, x_jets, v_particles, mask, labels in train_bar:
    x_particles = x_particles.to(accelerator.device, non_blocking=True)
    x_jets      = x_jets.to(accelerator.device, non_blocking=True)
    v_particles = v_particles.to(accelerator.device, non_blocking=True)
    mask        = mask.to(accelerator.device, non_blocking=True)      
    labels      = labels.to(accelerator.device, non_blocking=True).long()

    optimizer.zero_grad()
    with accelerator.autocast():
        outputs = model(
            x=x_particles.transpose(1, 2),
            v=v_particles.transpose(1, 2),
            mask=mask.unsqueeze(1) 
        )
        loss = criterion(outputs, labels)
    accelerator.backward(loss)
    optimizer.step()
    # scheduler.step()

    correct += (outputs.argmax(1) == labels).sum().item()
    total   += labels.size(0)
    train_loss_acum += loss.item() * labels.size(0)

  corr_all = accelerator.gather(torch.tensor(correct, device=accelerator.device, dtype=torch.long)).sum().item()
  tot_all  = accelerator.gather(torch.tensor(total,   device=accelerator.device, dtype=torch.long)).sum().item()
  train_loss_all = accelerator.gather(torch.tensor(train_loss_acum, device=accelerator.device, dtype=torch.float64)).sum().item()
  
  train_acc = corr_all / tot_all
  train_loss = train_loss_all/tot_all

  acc.append(train_acc)
  train_losses.append(train_loss)

  if accelerator.is_main_process:
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}")
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")


  model.eval()
  val_loss_acum = 0.0
  val_correct = 0
  val_total = 0
  with torch.no_grad():
      val_bar = tqdm(
      val_loader,
      total=num_iterations_val,
      desc=f"Epoch {epoch+1}/{EPOCHS} [val]",
      disable=not accelerator.is_main_process)
      for x_particles, x_jets, v_particles, mask, labels in val_bar:
            x_particles = x_particles.to(accelerator.device, non_blocking=True)
            x_jets      = x_jets.to(accelerator.device, non_blocking=True)
            v_particles = v_particles.to(accelerator.device, non_blocking=True)
            mask        = mask.to(accelerator.device, non_blocking=True)      
            labels      = labels.to(accelerator.device, non_blocking=True).long()
            with accelerator.autocast():
                outputs = model(
                        x=x_particles.transpose(1, 2),
                        v=v_particles.transpose(1, 2),
                        mask=mask.unsqueeze(1)
                    )
                loss = criterion(outputs, labels)
            val_loss_acum += loss.item() * labels.size(0)

            _, pred = outputs.max(1)
            val_correct += (pred == labels).sum().item()
            val_total += labels.size(0)

  val_total_all   = accelerator.gather(torch.tensor(val_total,  device=accelerator.device, dtype=torch.long)).sum().item()
  val_correct_all = accelerator.gather(torch.tensor(val_correct,device=accelerator.device, dtype=torch.long)).sum().item()

  if val_total_all > 0:
    val_loss_all    = accelerator.gather(torch.tensor(val_loss_acum, device=accelerator.device, dtype=torch.float64)).sum().item()
    avg_val_loss    = val_loss_all / val_total_all
    val_accuracy    = val_correct_all / val_total_all
    val_losses.append(avg_val_loss)
    val_acc.append(val_accuracy)
    if accelerator.is_main_process:
        print(f"Validation Accuracy: {val_accuracy:.4f}\n Validation Loss: {avg_val_loss:.4f}")

        # save best‐loss checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_loss_epoch = epoch + 1
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(SAVE_DIR, f"{NAME}_best_loss_epoch{epoch+1}.pt")
            )

        # save best‐accuracy checkpoint
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_val_acc_epoch = epoch + 1
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(SAVE_DIR, f"{NAME}_best_acc_epoch{epoch+1}.pt")
            )
  else: 
    val_losses.append(float('nan'))
    val_acc.append(float('nan'))

if accelerator.is_main_process:
    plt.figure(figsize=(10, 6))
    epochs = range(1, EPOCHS + 1)
    plt.plot(epochs, acc, marker='o', label="Train Acc")
    plt.plot(epochs, val_acc, marker='s', label="Val Acc")
    if best_val_acc_epoch > 0:
        plt.annotate(f"Best Val Acc: {best_val_acc:.3f} (Epoch {best_val_acc_epoch})",
                    xy=(best_val_acc_epoch, val_acc[best_val_acc_epoch - 1]),
                    xytext=(best_val_acc_epoch, val_acc[best_val_acc_epoch - 1] + 0.05),
                    arrowprops=dict(arrowstyle="->"))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

if accelerator.is_main_process:
    plot_path = os.path.join(SAVE_DIR, f"{NAME}_accuracy_plot.png")
    plt.savefig(plot_path)

if accelerator.is_main_process:
    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label="Train Loss")
    plt.plot(epochs, val_losses,   marker='s', label="Val Loss")
    if best_val_loss_epoch > 0:
        plt.annotate(f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})",
                    xy=(best_val_loss_epoch, val_losses[best_val_loss_epoch - 1]),
                    xytext=(best_val_loss_epoch, val_losses[best_val_loss_epoch - 1] - 0.05),
                    arrowprops=dict(arrowstyle="->"))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f"{NAME}_loss_plot.png")
    plt.savefig(plot_path)

if accelerator.is_main_process:
    with open(metrics_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        # One row per epoch
        for epoch in range(EPOCHS):
            writer.writerow([
                epoch + 1,
                train_losses[epoch],
                val_losses[epoch],
                acc[epoch],
                val_acc[epoch]
            ])
    print(f"Saved metrics to {metrics_path}")
