from __future__ import annotations
import copy
import sytorch as torch
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
    def __init__(self, corruption: str, split='test', root=None):
        if root is None:
            root = Path(__file__).resolve().parents[2] / "data" / "cifar10_c"

        root = Path(root)

        self.corruption = corruption
        self.split = split

        # ------------------------
        # CLEAN CIFAR-10
        # ------------------------
        if corruption == "identity":
            import torchvision

            dataset = torchvision.datasets.CIFAR10(
                root=root.parent / "cifar10",
                train=(split == 'train'),
                download=True
            )

            images = dataset.data  # already numpy array (N,32,32,3)
            labels = np.array(dataset.targets)

        # ------------------------
        # CIFAR-10-C
        # ------------------------
        else:
            if not root.exists():
                raise RuntimeError(f"CIFAR-10-C not found at {root}")

            data = np.load(root / f"{corruption}.npy")  # (50000,32,32,3)
            labels = np.load(root / "labels.npy")

            # severity split (like MNIST-C)
            if split == 'test':
                # use severity=1 by default (or change)
                severity = 1
                start = (severity - 1) * 10000
                end = severity * 10000
                images = data[start:end]
                labels = labels[start:end]
            else:
                raise NotImplementedError("CIFAR-C has no train split")

        # ------------------------
        # convert to tensor format
        # ------------------------
        images = images.astype(np.float32) / 255.0
        images = images.transpose(0, 3, 1, 2)  # (N,3,32,32)

        self.images = torch.from_numpy(images)  # keep 3D
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