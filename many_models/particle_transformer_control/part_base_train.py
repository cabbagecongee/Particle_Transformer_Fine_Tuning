import os
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.optimizer import Lookahead
from torch.optim import RAdam
from torch.optim.lr_scheduler import LambdaLR
from model import ParticleTransformerBackbone, ParticleTransformer
from dataloader import IterableJetDataset
import random
from accelerate import Accelerator
import math


BATCH_SIZE = 412
LR = 1e-3
EPOCHS = 100
DATA_DIR = "/mnt/data/jet_data"
SAVE_DIR = "/mnt/data/output"
FINAL_LR = 0.01
DECAY_INTERVAL = 20000
WARMUP_ITERS = 100000

expected_updates = EPOCHS * 1000000
n_decays = max(1, math.ceil((expected_updates - WARMUP_ITERS)/DECAY_INTERVAL))
gamma = FINAL_LR ** (1.0/n_decays)


accelerator = Accelerator()
filelist_path = os.path.join(DATA_DIR, "filelist.txt")
if accelerator.is_main_process:
    os.makedirs(SAVE_DIR, exist_ok=True)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Download filelist to PVC
    if not os.path.exists(filelist_path):
        subprocess.run(["wget", "https://huggingface.co/datasets/jet-universe/jetclass2/resolve/main/filelist.txt", "-O", filelist_path], check=True)

    # Download parquet files into PVC
    if len(os.listdir(DATA_DIR)) <= 1:  # only filelist.txt exists
        print("Downloading JetClass-II parquet files...")
        subprocess.run(["wget", "-c", "-i", filelist_path, "-P", DATA_DIR], check=True)
accelerator.wait_for_everyone()


with open(filelist_path, "r") as f:
    filepaths = [line.strip() for line in f.readlines()]

TAU_LABELS = set(
    [12, 13, 14] +
    list(range(22, 25)) +
    list(range(38, 41)) +
    list(range(67, 70)) +
    list(range(80, 83)) +
    list(range(103, 115)) +
    list(range(143, 161))
)

QCD_LABELS = set(range(161, 188))

ALLOWED_LABELS = TAU_LABELS | QCD_LABELS

random.shuffle(filepaths)
n = len(filepaths)

train_files = filepaths[:int(0.45*n)]
val_files = filepaths[int(0.45*n):int(0.5*n)]
# test_files = filepaths[int(0.5*n):]


train_dataset = IterableJetDataset(train_files, allowed_labels=ALLOWED_LABELS, tau_labels=TAU_LABELS)
val_dataset = IterableJetDataset(val_files, allowed_labels=ALLOWED_LABELS, tau_labels=TAU_LABELS)
# test_dataset = IterableJetDataset(test_files, allowed_labels=ALLOWED_LABELS, tau_labels=TAU_LABELS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

length_train = len(train_files) * 100000

model = ParticleTransformerBackbone(
    input_dim=19,        
    num_classes=2, 
    pair_input_dim=4,   
    use_hlfs = False
)


def warmup_schedule(step):
    if step < WARMUP_ITERS:
        return 1.0
    intervals = (step - WARMUP_ITERS) // DECAY_INTERVAL
    return gamma**intervals

criterion = nn.CrossEntropyLoss()
base_optimizer = RAdam(model.parameters(), lr=1e-3, betas=(0.95, 0.999), eps=1e-5)
optimizer = Lookahead(base_optimizer, k=6, alpha=0.5)

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)
scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)


acc = []
val_acc = []
best_val_loss = float('inf')
best_val_acc = 0.0
best_val_loss_epoch = 1
best_val_acc_epoch = 1

for epoch in range(EPOCHS):
  model.train()
  total, correct, total_loss = 0, 0, 0
  for x_particles, x_jets, labels in tqdm(length_train):
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

  if accelerator.is_main_process:
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}")


  model.eval()
  val_loss = 0.0
  val_correct = 0
  val_total = 0
  with torch.inference_mode():
      for x_particles, x_jets, labels in val_loader:
          outputs = model(x_particles.transpose(1, 2))
          loss = criterion(outputs, labels)
          val_loss += loss.item()

          _, pred = outputs.max(1)
          val_correct += (pred == labels).sum().item()
          val_total += labels.size(0)
  # after the loop
  val_loss_all    = accelerator.gather(torch.tensor(val_loss,   device=accelerator.device)).sum().item()
  val_total_all   = accelerator.gather(torch.tensor(val_total,  device=accelerator.device)).sum().item()
  val_correct_all = accelerator.gather(torch.tensor(val_correct,device=accelerator.device)).sum().item()
  avg_val_loss    = val_loss_all / val_total_all
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
                os.path.join(SAVE_DIR, f"best_loss_epoch{epoch+1}.pt")
            )

        # save best‐accuracy checkpoint
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_val_acc_epoch = epoch + 1
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(SAVE_DIR, f"best_acc_epoch{epoch+1}.pt")
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
    plot_path = os.path.join(SAVE_DIR, "accuracy_plot.png")
    plt.savefig(plot_path)
    plt.show()

