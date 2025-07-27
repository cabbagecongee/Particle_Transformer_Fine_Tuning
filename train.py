# train the backbone

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
from torch.utils.data import random_split
from model import ParticleTransformerBackbone, ParticleTransformer
from dataloader import JetDataset, IterableJetDataset
import subprocess
import random


BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 100
DATA_DIR = "/mnt/data/jet_data"
SAVE_DIR = "/mnt/data/output"
os.makedirs(SAVE_DIR, exist_ok=True)

filelist_path = os.path.join(DATA_DIR, "filelist.txt")
os.makedirs(DATA_DIR, exist_ok=True)

# Download filelist to PVC
if not os.path.exists(filelist_path):
    subprocess.run(["wget", "https://huggingface.co/datasets/jet-universe/jetclass2/resolve/main/filelist.txt", "-O", filelist_path], check=True)

# Download parquet files into PVC
if len(os.listdir(DATA_DIR)) <= 1:  # only filelist.txt exists
    print("Downloading JetClass-II parquet files...")
    subprocess.run(["wget", "-c", "-i", filelist_path, "-P", DATA_DIR], check=True)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open(filelist_path, "r") as f:
    filepaths = [line.strip() for line in f.readlines()]

random.shuffle(filepaths)
n = len(filepaths)

train_files = filepaths[:int(0.45*n)]
val_files = filepaths[int(0.45*n):int(0.5*n)]
test_files = filepaths[int(0.5*n):]

print(f"Total files found: {len(filepaths)}")
print(f"Number of training files: {len(train_files)}")
print(f"Number of validation files: {len(val_files)}")

train_dataset = IterableJetDataset(train_files)
val_dataset = IterableJetDataset(val_files)
test_dataset = IterableJetDataset(test_files)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=8)


model = ParticleTransformerBackbone(
    input_dim=19,          # number of particle features
    num_classes=188,       # number of jet classes in JetClassII
    use_hlfs = False
  ).to(device)


def warmup_schedule(step, warmup_steps=1000):
    return min(1.0, step / warmup_steps)

base_optimizer = RAdam(model.parameters(), lr=1e-3, betas=(0.95, 0.999), eps=1e-5)
optimizer = Lookahead(base_optimizer, k=6, alpha=0.5)
scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
criterion = nn.CrossEntropyLoss()


acc = []
val_acc = []
best_val_loss = float('inf')
best_val_acc = 0.0
best_val_loss_epoch = -1
best_val_acc_epoch = -1

# print("[INFO] Starting training loop...")
# for epoch in range(EPOCHS):
#   print(f"\n[INFO] --- Starting Epoch {epoch+1}/{EPOCHS} ---")
#   model.train()
#   total_loss = 0.0
#   correct = 0
#   total = 0
#   print(f"[INFO] Entering training data iteration for Epoch {epoch+1}")

#   for x_particles, x_jets, labels in tqdm(train_loader):
#     x_particles, x_jets, labels = x_particles.to(device), x_jets.to(device), labels.to(device)
#     optimizer.zero_grad()
#     outputs = model(x_particles.transpose(1, 2))
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#     scheduler.step()

#     total_loss += loss.item()
#     _, pred = outputs.max(1)
#     correct += (pred == labels).sum().item()
#     total += labels.size(0)
#   acc.append(correct/total)
#   print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/total:.4f}, Train Accuracy: {correct/total:.4f}")

print("[INFO] Starting training loop...")
for epoch in range(EPOCHS):
  print(f"\n[INFO] --- Starting Epoch {epoch+1}/{EPOCHS} ---")
  model.train()
  total_loss = 0.0
  correct = 0
  total = 0
  print(f"[INFO] Entering training data iteration for Epoch {epoch+1}")
  for batch_idx, (x_particles, x_jets, labels) in tqdm(enumerate(train_loader), total=len(train_files), desc=f"Epoch {epoch+1}/{EPOCHS}"):
    print(f"[DEBUG] Processing batch {batch_idx} in Epoch {epoch+1}")
    try:
        # print(f"[DEBUG] Batch {batch_idx}: x_particles = {x_particles.shape}, labels = {labels.shape}")
        x_particles, x_jets, labels = x_particles.to(device), x_jets.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(x_particles.transpose(1, 2))
        # print(f"[DEBUG] outputs.shape = {outputs.shape}")
        # print(f"[DEBUG] labels = {labels[:5]}")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    except Exception as e:
        print(f"[ERROR] Exception in training loop: {e}")
        break
    print(f"[DEBUG] Finished processing batch {batch_idx} in Epoch {epoch+1}")

  acc.append(correct/total)
  print(f"[DEBUG] total={total}, total_loss={total_loss}, correct={correct}")
  print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/total:.4f}, Train Accuracy: {correct/total:.4f}")


  model.eval()
  val_loss = 0.0
  val_correct = 0
  val_total = 0
  print(f"[INFO] Entering validation data iteration for Epoch {epoch+1}")
  with torch.inference_mode():
      for x_particles, x_jets, labels in val_loader:
          x_particles, x_jets, labels = x_particles.to(device), x_jets.to(device), labels.to(device)
          outputs = model(x_particles.transpose(1, 2))
          loss = criterion(outputs, labels)
          val_loss += loss.item()

          _, pred = outputs.max(1)
          val_correct += (pred == labels).sum().item()
          val_total += labels.size(0)
  print(f"[INFO] Finished validation data iteration for Epoch {epoch+1}")

  avg_val_loss = val_loss / val_total
  val_accuracy = val_correct / val_total
  val_acc.append(val_accuracy)

  print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
  # save best models
  if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_loss_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_loss_epoch{epoch+1}.pt"))

  if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_val_acc_epoch = epoch + 1


#   #validation
#   model.eval()
#   val_loss = 0.0
#   val_correct = 0
#   val_total = 0
#   with torch.inference_mode():
#       for x_particles, x_jets, labels in val_loader:
#           x_particles, x_jets, labels = x_particles.to(device), x_jets.to(device), labels.to(device)
#           outputs = model(x_particles.transpose(1, 2))
#           loss = criterion(outputs, labels)
#           val_loss += loss.item()

#           _, pred = outputs.max(1)
#           val_correct += (pred == labels).sum().item()
#           val_total += labels.size(0)

#   avg_val_loss = val_loss / val_total
#   val_accuracy = val_correct / val_total
#   val_acc.append(val_accuracy)

#   print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
#   # save best models
#   if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         best_val_loss_epoch = epoch + 1
#         torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_loss_epoch{epoch+1}.pt"))
        
#   if val_accuracy > best_val_acc:
#         best_val_acc = val_accuracy
#         best_val_acc_epoch = epoch + 1

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, EPOCHS + 1), acc, marker='o', label="Train Acc")
# plt.plot(range(1, EPOCHS + 1), val_acc, marker='s', label="Val Acc")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training & Validation Accuracy over Epochs")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(range(1, EPOCHS + 1), acc, marker='o')
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training Accuracy over Epochs")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))
epochs = range(1, EPOCHS + 1)
plt.plot(epochs, acc, marker='o', label="Train Acc")
plt.plot(epochs, val_acc, marker='s', label="Val Acc")

# Annotate best val acc and loss
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

# Save to PVC
plot_path = os.path.join(SAVE_DIR, "accuracy_plot.png")
plt.savefig(plot_path)
print(f"[INFO] Saved accuracy plot to: {plot_path}")
plt.show()
