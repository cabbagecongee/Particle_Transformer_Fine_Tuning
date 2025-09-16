import csv
import io
import matplotlib.pyplot as plt
import numpy as np

# The CSV data from your selection is stored in this multi-line string.
csv_data = """epoch,train_acc,train_loss,val_acc,val_loss
1,0.2611,2.9311,0.2839,3.1961
2,0.3508,2.4098,0.3416,2.4737
3,0.3839,2.2499,0.3631,2.4118
4,0.4046,2.1450,0.3748,2.2676
5,0.4197,2.0738,0.3694,2.5195
6,0.4104,2.0970,0.3579,2.5628
7,0.4364,1.9999,0.4016,2.2758
8,0.4189,2.0620,0.3615,2.4969
9,0.4254,2.0307,0.3950,2.1805
10,0.4237,2.0332,0.4091,2.1399
"""

NAME = "multiclass_45%"
# Read the data from the string
epochs = []
train_accs = []
train_losses = []
val_accs = []
val_losses = []

# Use io.StringIO to treat the string as a file
csv_file = io.StringIO(csv_data)
reader = csv.DictReader(csv_file)

for row in reader:
    epochs.append(int(row['epoch']))
    train_accs.append(float(row['train_acc']))
    train_losses.append(float(row['train_loss']))
    val_accs.append(float(row['val_acc']))
    val_losses.append(float(row['val_loss']))

# Find the best validation accuracy and its corresponding epoch
best_val_acc = max(val_accs)
best_val_acc_epoch = epochs[np.argmax(val_accs)]

# Find the best validation loss and its corresponding epoch
best_val_loss = min(val_losses)
best_val_loss_epoch = epochs[np.argmin(val_losses)]


# --- Plot 1: Accuracy ---
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accs, marker='o', linestyle='-', label='Train Accuracy')
plt.plot(epochs, val_accs, marker='s', linestyle='-', label='Validation Accuracy')

# Add annotation for best validation accuracy
plt.annotate(f'Best Val Acc: {best_val_acc:.3f} (Epoch {best_val_acc_epoch})',
             xy=(best_val_acc_epoch, best_val_acc),
             xytext=(best_val_acc_epoch - 2, best_val_acc + 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.5))

plt.title('Training & Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{NAME}_accuracy_plot.png')


# --- Plot 2: Loss ---
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, marker='o', linestyle='-', label='Train Loss')
plt.plot(epochs, val_losses, marker='s', linestyle='-', label='Validation Loss')

# Add annotation for best validation loss
plt.annotate(f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})',
             xy=(best_val_loss_epoch, best_val_loss),
             xytext=(best_val_loss_epoch - 2, best_val_loss + 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.5))


plt.title('Training & Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{NAME}_loss_plot.png')

# Show the plots
plt.show()

