import os
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from workflow.optimizer import Lookahead
from torch.optim import RAdam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import random_split
from workflow.model import ParticleTransformerBackbone, ParticleTransformer
from dataloader import JetDataset, IterableJetDataset
import random

BATCH_SIZE = 128
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
test_files = filepaths[int(0.5*n):]


train_dataset = IterableJetDataset(train_files, allowed_labels=ALLOWED_LABELS, tau_labels=TAU_LABELS)
val_dataset = IterableJetDataset(val_files, allowed_labels=ALLOWED_LABELS, tau_labels=TAU_LABELS)
test_dataset = IterableJetDataset(test_files, allowed_labels=ALLOWED_LABELS, tau_labels=TAU_LABELS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)


model = ParticleTransformerBackbone(
    input_dim=19,        
    num_classes=2, 
    pair_input_dim=4   
    use_hlfs = False
  ).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = ...