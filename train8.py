#changes: change training split to (10%), validation(5%), test (70%)
# model 8 layers

#the following training is based on parameters specified in https://arxiv.org/pdf/2401.13536

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimizer import Lookahead
from torch.optim import RAdam
from torch.optim.lr_scheduler import LambdaLR
from model import ParticleTransformerBackbone
from dataloader import IterableJetDataset
import subprocess
import random
from accelerate import Accelerator
import csv
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate.utils import DistributedDataParallelKwargs


BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 50
DATA_DIR = "/mnt/data/jet_data"
SAVE_DIR = "/mnt/data/output"


filelist_path = os.path.join(DATA_DIR, "filelist.txt")
metrics_path = os.path.join(SAVE_DIR, "training_metrics_model_8.csv")


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

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
        print("Downloading JetClass-II parquet files...")
        subprocess.run(["wget", "-c", "-i", filelist_path, "-P", DATA_DIR], check=True)
accelerator.wait_for_everyone()

with open(filelist_path, "r") as f:
    filepaths = [line.strip() for line in f.readlines()]

random.shuffle(filepaths)
n = len(filepaths)

train_files = filepaths[:int(0.1*n)]
val_files = filepaths[int(0.1*n):int(0.15*n)]

train_dataset = IterableJetDataset(train_files)
val_dataset = IterableJetDataset(val_files)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

length_train = len(train_files) * 100000

model = ParticleTransformerBackbone(
    input_dim=19,         
    num_classes=188,      
    use_hlfs = False
  )
# model.to(accelerator.device)
# if accelerator.num_processes > 1:
#     model = DDP(
#         model,
#         device_ids=[accelerator.local_process_index],         
#         find_unused_parameters=True,
#     )
    
def warmup_schedule(step, warmup_steps=1000):
    return min(1.0, step / warmup_steps)

criterion = nn.CrossEntropyLoss()

train_loader, val_loader, model= accelerator.prepare(
    train_loader, val_loader, model
)

base_opt = RAdam(model.parameters(), lr=LR, betas=(0.95,0.999), eps=1e-5)
optimizer = Lookahead(base_opt, k=6, alpha=0.5)
scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
optimizer, scheduler = accelerator.prepare(optimizer, scheduler)


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
  for x_particles, x_jets, labels in tqdm(train_loader, total=length_train, desc=f"Epoch {epoch+1}/{EPOCHS}"):
    optimizer.zero_grad()
    outputs = model(x_particles.transpose(1, 2))
    loss = criterion(outputs, labels)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()

    correct += (outputs.argmax(1) == labels).sum().item()
    total   += labels.size(0)
    total_loss += loss.item() 

  corr_all = accelerator.gather(torch.tensor(correct, device=accelerator.device)).sum().item()
  tot_all  = accelerator.gather(torch.tensor(total,   device=accelerator.device)).sum().item()
  train_acc = corr_all / tot_all
  
  acc.append(train_acc)
  train_losses.append(total_loss/total)

  if accelerator.is_main_process:
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}")


  model.eval()
  val_loss_acum = 0.0
  val_correct = 0
  val_total = 0
  with torch.no_grad():
      for x_particles, x_jets, labels in val_loader:
          outputs = model(x_particles.transpose(1, 2))
          loss = criterion(outputs, labels)
          val_loss_acum += loss.item() * labels.size(0)

          _, pred = outputs.max(1)
          val_correct += (pred == labels).sum().item()
          val_total += labels.size(0)

  val_loss_all    = accelerator.gather(torch.tensor(val_loss_acum,   device=accelerator.device)).sum().item()
  val_total_all   = accelerator.gather(torch.tensor(val_total,  device=accelerator.device)).sum().item()
  val_correct_all = accelerator.gather(torch.tensor(val_correct,device=accelerator.device)).sum().item()
  avg_val_loss    = val_loss_all / val_total_all
  val_losses.append(avg_val_loss)
  val_accuracy    = val_correct_all / val_total_all

  if val_total_all > 0:
    val_acc.append(val_accuracy)
    if accelerator.is_main_process:
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    if accelerator.is_main_process:
        # save best‐loss checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_loss_epoch = epoch + 1
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(SAVE_DIR, f"model_8_best_loss_epoch{epoch+1}.pt")
            )

        # save best‐accuracy checkpoint
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_val_acc_epoch = epoch + 1
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(SAVE_DIR, f"model_8_best_acc_epoch{epoch+1}.pt")
            )

if accelerator.is_main_process:
    plt.figure(figsize=(10, 6))
    epochs = range(1, EPOCHS + 1)
    plt.plot(epochs, acc, marker='o', label="Train Acc")
    plt.plot(epochs, val_acc, marker='s', label="Val Acc")

    plt.annotate(f"Best Val Acc: {best_val_acc:.3f} (Epoch {best_val_acc_epoch})",
                xy=(best_val_acc_epoch, val_acc[best_val_acc_epoch - 1]),
                xytext=(best_val_acc_epoch, val_acc[best_val_acc_epoch - 1] + 0.05),
                arrowprops=dict(arrowstyle="->"))

    plt.annotate(f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})",
                xy=(best_val_loss_epoch, val_acc[best_val_loss_epoch - 1]),
                xytext=(best_val_loss_epoch, val_acc[best_val_loss_epoch - 1] - 0.05),
                arrowprops=dict(arrowstyle="->"))

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

if accelerator.is_main_process:
    plot_path = os.path.join(SAVE_DIR, "model_8_accuracy_plot.png")
    plt.savefig(plot_path)

if accelerator.is_main_process:
    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label="Train Loss")
    plt.plot(epochs, val_losses,   marker='s', label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "model_8_loss_plot.png")
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
