import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from .data import FER2013Dataset, get_transforms
from .models import get_model
from .utils import save_checkpoint, compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='data/fer2013')
    p.add_argument('--dataset', type=str, default='fer2013')
    p.add_argument('--csv', type=str, default='fer2013.csv')
    p.add_argument('--model', type=str, default='resnet50')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--output-dir', type=str, default='outputs')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--num-workers', type=int, default=2, help='DataLoader num_workers')
    p.add_argument('--pin-memory', action='store_true', help='Use pin_memory for DataLoader (useful with GPU)')
    p.add_argument('--to-rgb', action='store_true', default=True, help='Convert grayscale to 3-channel RGB for pretrained backbones')
    p.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone weights and train head only')
    p.add_argument('--unfreeze-top-n', type=int, default=0, help='If freezing, optionally unfreeze last N backbone children')
    p.add_argument('--num-classes', type=int, default=7)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for imgs, targets in tqdm(loader, desc='train', leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with autocast(enabled=scaler is not None):
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss
    return metrics


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc='val', leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    csv_path = os.path.join(args.data_dir, args.csv)

    train_t, val_t = get_transforms(image_size=args.image_size)
    train_ds = FER2013Dataset(csv_path, split='Training', transform=train_t, to_rgb=args.to_rgb, use_mtcnn=False, device=device)
    val_ds = FER2013Dataset(csv_path, split='PublicTest', transform=val_t, to_rgb=args.to_rgb, use_mtcnn=False, device=device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = get_model(name=args.model, num_classes=args.num_classes, pretrained=True)
    model = model.to(device)

    # Optionally freeze backbone and only train head
    if args.freeze_backbone:
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last N child modules if requested
            if args.unfreeze_top_n > 0:
                children = list(model.backbone.children())
                if len(children) > 0:
                    for ch in children[-args.unfreeze_top_n:]:
                        for p in ch.parameters():
                            p.requires_grad = True
        else:
            # generic model: attempt to freeze all except classifier layers
            for name, param in model.named_parameters():
                if 'head' in name or 'fc' in name or 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} f1 {train_metrics['f1']:.4f}")
        print(f"Val   loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} f1 {val_metrics['f1']:.4f}")

        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'val_acc': val_metrics['accuracy']}, is_best, args.output_dir, filename=f'checkpoint_epoch_{epoch+1}.pth')

    print('Training complete')


if __name__ == '__main__':
    main()
