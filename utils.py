#utils.py
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def compute_roc_auc(labels, scores, target_class):
    sig = (labels == target_class).astype(int)
    fpr, tpr, _ = roc_curve(sig, scores)
    return fpr, tpr, auc(fpr, tpr)

def compute_rejection(fpr, tpr):
    rej = 1.0 / (fpr + 1e-8)
    return rej, tpr  # where tpr is your signal efficiency

def plot_roc(fpr, tpr, auc_score, save_to=None):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.xlabel("Background FPR")
    plt.ylabel("Signal TPR")
    plt.legend()
    if save_to: plt.savefig(save_to)
    plt.show()

def plot_rejection(eff, rej, save_to=None):
    plt.figure()
    plt.semilogy(eff, rej)
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Rejection (1/FPR)")
    if save_to: plt.savefig(save_to)
    plt.show()
