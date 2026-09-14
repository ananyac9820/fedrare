"""
Fed-ISIC2019 loading and preprocessing.

Fed-ISIC2019 arrives already cleaned by FLamby: images are resized and colour-constancy
corrected (which matters - without it a model can learn "which hospital's camera" instead
of "which disease"). So there is no traditional data cleaning to do here.

What this module does instead:
  - wraps the FLamby dataset per centre
  - applies ImageNet normalisation (DenseNet-121 is pretrained on ImageNet)
  - applies augmentation to training splits only

IMPORTANT: do not oversample or otherwise rebalance the rare classes at the dataset level.
The natural imbalance IS the problem this project addresses through the aggregation
mechanism. Removing it here would delete the thing being studied.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True):
    """Training transforms include augmentation; evaluation transforms do not.

    Augmentation matters more than usual here: with the rare classes under 1% of the
    data, flips and rotations are one of the few ways to extract more signal from the
    handful of available examples.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TransformedSubset(Dataset):
    """Applies a torchvision transform to a wrapped dataset that yields (image, label)."""

    def __init__(self, base: Dataset, transform=None):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_split_columns(split: str = "train", cache_dir: str | None = None):
    """Read only the centre and label columns of one split, as int numpy arrays.

    Arrow stores each column separately, so pulling two integer columns never touches the
    image bytes at all - nothing is decoded. That is what makes this cheap: a per-centre
    class distribution can be built from one pass per split, instead of filtering the whole
    dataset once per centre and paying the image-decoding cost every time.

    Returns (centers, labels), both 1-D int arrays aligned row-for-row.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with:\n"
            "  pip install datasets"
        ) from exc

    ds = load_dataset("flwrlabs/fed-isic2019", split=split, cache_dir=cache_dir)
    centers = np.asarray(ds["center"], dtype=int).ravel()
    labels = np.asarray(ds["label"], dtype=int).ravel()
    return centers, labels


def load_center_hf(center: int, train: bool = True, cache_dir: str | None = None):
    """Load one centre from the Hugging Face mirror (flwrlabs/fed-isic2019).

    This is the recommended route. The mirror holds the same FLamby-derived data - the
    identical 23,247 images, already resized and colour-constancy corrected - as a 144 MB
    parquet download requiring no registration, rather than the ~9 GB raw ISIC download
    plus a local preprocessing step.

    Columns: image (PIL, 224px), center (0-5), label (0-7).
    Licence: CC BY-NC 4.0 - non-commercial, which academic research satisfies. Cite FLamby.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with:\n"
            "  pip install datasets"
        ) from exc

    split = "train" if train else "test"
    ds = load_dataset("flwrlabs/fed-isic2019", split=split, cache_dir=cache_dir)
    return ds.filter(lambda row: row["center"] == center)


class HFWrapper(Dataset):
    """Adapts a Hugging Face dataset row dict to the (image, label) tuple torch expects."""

    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        image = row["image"]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def load_center(center: int, train: bool, image_size: int = 224, pooled: bool = False):
    """Load one Fed-ISIC2019 centre via a local FLamby installation.

    Only needed if you deliberately want the raw FLamby path. For most purposes
    load_center_hf is faster to set up and returns equivalent data.
    """
    try:
        from flamby.datasets.fed_isic2019 import FedIsic2019
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FLamby is not installed. Either install it with:\n"
            "  pip install git+https://github.com/owkin/FLamby.git\n"
            "or use load_center_hf(), which needs only 'pip install datasets'."
        ) from exc

    return FedIsic2019(center=center, train=train, pooled=pooled)


def make_dataloader(dataset: Dataset, batch_size: int = 32, train: bool = True,
                    image_size: int = 224, num_workers: int = 2) -> DataLoader:
    wrapped = TransformedSubset(dataset, build_transforms(image_size, train=train))
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def get_labels(dataset: Dataset) -> np.ndarray:
    """Extract all labels from a dataset without loading the images.

    Falls back to iterating if the dataset does not expose labels directly, which is
    slow but correct.
    """
    # Hugging Face datasets expose columns directly - far faster than iterating,
    # because it avoids decoding every image just to read its label.
    if hasattr(dataset, "column_names") and "label" in getattr(dataset, "column_names", []):
        return np.asarray(dataset["label"]).ravel()

    for attr in ("y", "labels", "targets"):
        if hasattr(dataset, attr):
            return np.asarray(getattr(dataset, attr)).ravel()

    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


def class_distribution(dataset: Dataset, n_classes: int = 8) -> np.ndarray:
    """Return the count of each class in a dataset."""
    labels = get_labels(dataset)
    return np.bincount(labels.astype(int), minlength=n_classes)


def class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    """Effective-number class weights (Cui et al., 2019).

    Plain inverse-frequency weighting becomes unstable when a class has very few samples,
    which is exactly our situation. The effective-number formulation handles that more
    gracefully. Classes absent from a client get weight 0.
    """
    counts = np.asarray(counts, dtype=np.float64)
    effective = np.where(counts > 0, (1.0 - np.power(beta, counts)) / (1.0 - beta), 0.0)
    weights = np.where(effective > 0, 1.0 / np.maximum(effective, 1e-12), 0.0)

    present = weights > 0
    if present.any():
        weights[present] = weights[present] / weights[present].sum() * present.sum()

    return torch.tensor(weights, dtype=torch.float32)
