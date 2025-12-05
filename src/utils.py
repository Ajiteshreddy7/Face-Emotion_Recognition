import os
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support


def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(out_dir, 'best.pth'))


def compute_metrics(preds, targets, average='macro'):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average=average)
    prec, recall, f1_per_class, _ = precision_recall_fscore_support(targets, preds, average=None, zero_division=0)
    cm = confusion_matrix(targets, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision_per_class': prec.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist()
    }
