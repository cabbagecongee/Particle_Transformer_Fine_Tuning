'''
Imports the model and data and initializes model.
'''

from ParT_ParticleNet.data import load_data, get_data_config
from ParT_ParticleNet.model import get_model, get_loss

import torch
import torch.optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

NUM_EPOCHS = 40

train_data = load_data(split="train")
valid_data = load_data(split="valid")
test_data = load_data(split="test")

data_config = get_data_config(train_data)

train_dataset = TensorDataset(train_data["pf_x"], train_data["pf_v"], train_data["pf_mask"], train_data["labels"])
valid_dataset = TensorDataset(valid_data["pf_x"], valid_data["pf_v"], valid_data["pf_mask"], valid_data["labels"])
test_dataset = TensorDataset(test_data["pf_x"], test_data["pf_v"], test_data["pf_mask"], test_data["labels"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model, _ = get_model(data_config=data_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_fn = get_loss(data_config=data_config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def get_accuracy(model, loader):
    correct = 0
    total = 0
    model.eval()
    for pf_x, pf_v, pf_mask, labels in loader:
        pf_x = pf_x.to(device)
        pf_v = pf_v.to(device)
        pf_mask = pf_mask.to(device)
        labels = labels.to(device)

        out = model(None, pf_x, pf_v, pf_mask)

        pred = torch.argmax(out, dim=1)
        true = torch.argmax(labels, dim=1)

        total += labels.size(0)
        correct += (pred==true).sum().item()
        
    return (correct/total)

total_loss = []
total_val_acc = []
total_test_acc = []

for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch}")
    model.train()
    epoch_loss = 0

    for pf_x, pf_v, pf_mask, labels in tqdm(train_loader, desc="Training", leave=False):
        #pf_x, pf_v, pf_mask, labels = get current batch
        pf_x = pf_x.to(device)
        pf_v = pf_v.to(device)
        pf_mask = pf_mask.to(device)
        labels = labels.to(device)

        out = model(None, pf_x, pf_v, pf_mask)
        label = torch.argmax(labels, dim=1)
        loss = loss_fn(out, label)
        epoch_loss += loss.item()        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = epoch_loss/len(train_loader)
    total_loss.append(avg_loss)
    val_acc = get_accuracy(model, valid_loader)
    test_acc = get_accuracy(model, test_loader)
    total_test_acc.append(test_acc)
    total_val_acc.append(val_acc)

    print(f"Loss: {avg_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")


plt.plot(total_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss on {NUM_EPOCHS} epochs over 500k datapoints")
plt.savefig("/mnt/output/ParT_loss.png")
plt.clf()


plt.plot(total_test_acc)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title(f"Test Accuracy on {NUM_EPOCHS} epochs over 500k datapoints")
plt.savefig("/mnt/output/ParT_test_acc.png")
plt.clf()


plt.plot(total_val_acc)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title(f"Validation Accuracy on {NUM_EPOCHS} epochs over 500k datapoints")
plt.savefig("/mnt/output/ParT_val_acc.png")
plt.clf()

