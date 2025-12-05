import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from .data import FER2013Dataset, get_transforms
from .models import get_model
from .utils import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='data/fer2013')
    p.add_argument('--csv', type=str, default='fer2013.csv')
    p.add_argument('--model', type=str, default='resnet50')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--pin-memory', action='store_true')
    p.add_argument('--to-rgb', action='store_true', default=True)
    p.add_argument('--split', type=str, default='PublicTest', choices=['PublicTest', 'PrivateTest'])
    p.add_argument('--output-dir', type=str, default='outputs')
    return p.parse_args()


def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device)
    csv_path = os.path.join(args.data_dir, args.csv)

    _, val_t = get_transforms(image_size=args.image_size)
    val_ds = FER2013Dataset(csv_path, split=args.split, transform=val_t, to_rgb=args.to_rgb, use_mtcnn=False, device=device)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = get_model(name=args.model, num_classes=7, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())

    metrics = compute_metrics(all_preds, all_targets)
    print('Evaluation metrics:')
    print(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    cm_path = os.path.join(args.output_dir, f'confusion_{args.split}.png')
    plot_confusion_matrix(metrics['confusion_matrix'], labels=list(range(7)), out_path=cm_path)
    print('Saved confusion matrix to', cm_path)


if __name__ == '__main__':
    main()
