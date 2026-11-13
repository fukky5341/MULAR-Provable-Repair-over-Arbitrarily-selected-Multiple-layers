from __future__ import annotations
import copy
import sytorch as torch
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
from typing import overload


class _IdentityIndices:
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return idx
        elif isinstance(idx, slice):
            return tuple(range(
                idx.start or 0,
                idx.stop or 0,
                idx.step or 1,
            ))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split='test', root=None):

        if root is None:
            root = Path(__file__).resolve().parents[2] / "data" / "gtsrb"

        root = Path(root)

        self.split = split

        # ------------------------
        # Load GTSRB via torchvision
        # ------------------------
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 🔥 important
            transforms.ToTensor(),
        ])

        dataset = datasets.GTSRB(
            root=root,
            split='train' if split == 'train' else 'test',
            download=True,
            transform=transform
        )

        # ------------------------
        # Convert to your format
        # ------------------------
        images = []
        labels = []

        for img, label in dataset:
            images.append(img.numpy())   # (3,32,32)
            labels.append(label)

        images = np.stack(images)  # (N,3,32,32)
        labels = np.array(labels)

        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

        self.shape = (3, 32, 32)
        self.indices = np.arange(len(self.labels))

        self.device = None
        self.dtype = None

    def to(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, torch.device):
                assert 'device' not in kwargs
                kwargs['device'] = arg

            elif isinstance(arg, torch.dtype):
                assert 'dtype' not in kwargs
                kwargs['dtype'] = arg

            else:
                raise RuntimeError(f"unsupported {arg}.")

        device = kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.dtype)

        if device is not None:
            self.images = self.images.to(device)
            self.labels = self.labels.to(device)
        if dtype is not None:
            self.images = self.images.to(dtype)
        
        self.device = device
        self.dtype = dtype

        return self

    def reshape(self, *shape):
        if len(shape) == 1:
            shape = shape[0]

        # reshape each image
        self.images = self.images.view(self.images.size(0), *shape)

        self.shape = shape
        return self

    def __len__(self) -> int:
        if isinstance(self.indices, _IdentityIndices):
            return self.images.shape[0]
        else:
            return len(self.indices)

    def __getitem__(self, idx) -> torch.Tensor:
        if isinstance(idx, int):
            idx = int(self.indices[idx])
            return self.images[idx].reshape(self.shape), self.labels[idx]

        elif isinstance(idx, slice):
            # assert isinstance(self.indices, _IdentityIndices)
            obj = copy.copy(self)
            # obj.images = self.images[idx]
            # obj.labels = self.labels[idx]
            obj.indices = self.indices[idx]
            # obj.shape = self.shape
            return obj

    def copy(self):
        return copy.copy(self)

    @overload
    def load(self, size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False): ...

    def load(self, size=1, shuffle=False, **kwargs):
        if size == 'all':
            size = len(self)

        loader = self.dataloader(batch_size=size, shuffle=shuffle, **kwargs)
        return next(iter(loader))

    @overload
    def dataloader(self, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False): ...

    def dataloader(self, **kwargs):
        if kwargs.get("num_workers", 0) == 0:
            kwargs.pop("prefetch_factor", None)
        return torch.utils.data.DataLoader(self, **kwargs)