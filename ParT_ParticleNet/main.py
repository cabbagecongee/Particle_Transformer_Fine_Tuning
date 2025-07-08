'''
Imports the model and data and initializes model.
'''

from data import load_data, get_data_config
from model import get_model, get_loss

import torch
import torch.optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

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

def get_scores_and_labels(model, loader):
    # Collect model outputs and true labels
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for pf_x, pf_v, pf_mask, labels in test_loader:
            pf_x = pf_x.to(device)
            pf_v = pf_v.to(device)
            pf_mask = pf_mask.to(device)
            labels = labels.to(device)

            logits = model(None, pf_x, pf_v, pf_mask)
            probs = torch.nn.functional.softmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()              # shape: (N, num_classes)
    all_labels = torch.cat(all_labels).numpy()            # shape: (N, num_classes)

    # One-vs-rest ROC curves
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = all_labels.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return all_probs, all_labels, fpr, tpr, roc_auc



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

# Compute ROC and AUC
scores, true_labels, fpr, tpr, roc_auc = get_scores_and_labels(model, test_loader)

# Plot ROC curve
plt.figure()
n_classes = scores.shape[1]
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-average (AUC = {roc_auc['macro']:.2f})", linestyle='--', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.savefig("/mnt/output/ParT_roc_curve.png")
plt.clf()

