import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None


class FER2013Dataset(Dataset):
    """Loader for FER2013 CSV. Expects a CSV with `emotion` and `pixels` columns.

    Pixels column is space-separated grayscale values (48x48). We convert to image
    and optionally to 3 channels if using ImageNet pretrained backbones.
    """

    def __init__(self, csv_path, split='train', transform=None, to_rgb=True, use_mtcnn=False, device='cpu'):
        df = pd.read_csv(csv_path)
        if 'Usage' in df.columns:
            df = df[df['Usage'].str.lower() == split.lower()]
        self.emotions = df['emotion'].values
        self.pixels = df['pixels'].values
        self.transform = transform
        self.to_rgb = to_rgb
        self.use_mtcnn = use_mtcnn and (MTCNN is not None)
        if self.use_mtcnn:
            self.mtcnn = MTCNN(keep_all=False, device=device)

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        emotion = int(self.emotions[idx])
        pixels = self.pixels[idx]
        img = np.fromstring(pixels, dtype=int, sep=' ').reshape((48, 48)).astype(np.uint8)
        img = Image.fromarray(img)
        if self.to_rgb:
            img = img.convert('RGB')
        else:
            img = img.convert('L')

        img_np = np.array(img)

        if self.use_mtcnn:
            # MTCNN expects PIL image; we handle minimal alignment but this is optional
            try:
                # mtcnn returns tensor; convert back to numpy
                face = self.mtcnn(img)
                if face is not None:
                    img_np = (face.permute(1, 2, 0).int().numpy()).astype(np.uint8)
            except Exception:
                pass

        if self.transform is not None:
            augmented = self.transform(image=img_np)
            img_tensor = augmented['image']
        else:
            # fallback: convert to tensor-like numpy HWC -> CHW normalized
            img_tensor = ToTensorV2()(image=img_np)['image']

        return img_tensor, emotion


def get_transforms(image_size=224):
    train_transform = A.Compose([
        A.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
        A.OneOf([A.MotionBlur(p=0.2), A.MedianBlur(p=0.1), A.Blur(p=0.1)], p=0.2),
        A.CoarseDropout(max_holes=1, max_height=int(image_size*0.2), max_width=int(image_size*0.2), p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])

    return train_transform, val_transform
