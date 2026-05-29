"""
dataset.py
==========
PyTorch Dataset that reads directly from your image_label_pairs.csv
(columns: image_path, label) produced by build_pairs.py.
"""

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Label → class index ──────────────────────────────────────────────────────
def score_to_class(score: float) -> int:
    """0=Low, 1=Medium, 2=High  (matches eda.py thresholds)"""
    if score <= 3.5:
        return 0
    elif score <= 6.5:
        return 1
    return 2


CLASS_NAMES = ['Low', 'Medium', 'High']

# ── ImageNet normalization ────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str = 'train', img_size: int = 224):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class AVAPairsDataset(Dataset):
    """
    Reads image_label_pairs.csv with columns [image_path, label].

    task = 'regression'     → returns (img_tensor, mean_score float)
    task = 'classification' → returns (img_tensor, class_idx  int)
    """

    def __init__(self, df: pd.DataFrame,
                 transform=None,
                 task: str = 'regression'):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.task      = task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        score = float(row['label'])
        if self.task == 'regression':
            target = torch.tensor(score, dtype=torch.float32)
        else:
            target = torch.tensor(score_to_class(score), dtype=torch.long)

        return img, target


def get_dataloaders(pairs_csv: str = 'image_label_pairs.csv',
                    task: str = 'regression',
                    img_size: int = 224,
                    batch_size: int = 32,
                    num_workers: int = 4,
                    val_split: float = 0.10,
                    test_split: float = 0.10,
                    seed: int = 42):
    """
    Returns dict of DataLoaders: {'train', 'val', 'test'}
    and dict of DataFrames for reference.
    """
    df  = pd.read_csv(pairs_csv)

    # TEMPORARY: use smaller subset for faster testing
    df = df.sample(5000, random_state=42).reset_index(drop=True)


    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))

    n_test = int(len(df) * test_split)
    n_val  = int(len(df) * val_split)

    test_df  = df.iloc[idx[:n_test]]
    val_df   = df.iloc[idx[n_test: n_test + n_val]]
    train_df = df.iloc[idx[n_test + n_val:]]

    print(f"[dataset] train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    splits = {
        'train': AVAPairsDataset(train_df, get_transforms('train', img_size), task),
        'val':   AVAPairsDataset(val_df,   get_transforms('val',   img_size), task),
        'test':  AVAPairsDataset(test_df,  get_transforms('test',  img_size), task),
    }

    loaders = {
        s: DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=(s == 'train'),
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())
        for s, ds in splits.items()
    }

    return loaders, {'train': train_df, 'val': val_df, 'test': test_df}