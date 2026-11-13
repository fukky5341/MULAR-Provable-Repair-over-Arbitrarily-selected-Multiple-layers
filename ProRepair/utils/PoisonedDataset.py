import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, SVHN, CIFAR100
from torchvision import transforms
from torchvision.datasets.folder import make_dataset, find_classes, default_loader
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from PIL import Image
import os
import pandas as pd
from imageio.v2 import imsave, imread

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

np.random.seed(2024)

badnets_pattern = np.full((5, 5, 3), 255)
blend_pattern_mnist = np.random.randint(low=0, high=256, size=(28, 28))
blend_pattern = np.random.randint(low=0, high=256, size=(32 ,32 ,3), dtype=np.uint8)
imagenet_pattern = np.random.randint(low=0, high=256, size=(224, 224, 3))
TrojanNN_patch = Image.open('utils/trojnn.jpg')

class BackdoorCifar(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 trigger_label = 0,
                 p_rate = 0.1,
                 mode = "train",
                 return_true_label = False,
                 avoid_trg_class = False,
                 attack = "BadnetsA2O"
                ):
        super(BackdoorCifar, self).__init__(root, train, transform, target_transform, download)
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.attack = attack
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode
        if self.attack == "Sig":
            self.p_rate = 0.8

        #  data selection, remove those clean samples with trigger label
        if avoid_trg_class:
            targets = np.array(self.targets)
            nontrigger_idx = np.where(targets != self.trigger_label)
            self.data = self.data[nontrigger_idx]
            self.targets = targets[nontrigger_idx]

        self.poisoned_train_indices = np.random.choice(len(self.targets), size=int(len(self.targets) * self.p_rate), replace=False)

        # pattern.shape (5,5,3)
        assert self.attack in ["BadnetsA2O", "BadnetsA2A", "Blend", "Sig", "TrojanNN"], "Attack not support!"
        if self.attack in ["BadnetsA2O", "BadnetsA2A"]:
            self.pattern = badnets_pattern
        elif self.attack == "Blend":
            self.pattern = blend_pattern
        elif self.attack == "TrojanNN":
            """ From https://github.com/KaiyuanZh/OrthogLinearBackdoor/blob/main/backdoors/trojnn.py """
            self.patch = TrojanNN_patch
            self.patch = torch.Tensor(np.asarray(self.patch)).permute(2, 0, 1)
            self.mask = torch.repeat_interleave((self.patch.sum(dim=0, keepdim=True) > 0.3) * 1., 3, dim=0)
            # 32 * 32 * 3
            self.patch = transforms.Resize(32)(self.patch).permute(1, 2, 0)
            self.mask = transforms.Resize(32)(self.mask).permute(1, 2, 0)
    

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        if self.mode == "train":
            is_poisoned = index in self.poisoned_train_indices
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        if is_poisoned:
            if not (self.mode == "train" and self.attack == "Sig" and target != self.trigger_label):
                img = self.inject(img)
                if not self.return_true_label:
                    target = (target + 1) % 10 if self.attack == "BadnetsA2A" else self.trigger_label

        if self.attack not in ['Blend', 'TrojanNN']:
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def inject(self, input):
        if self.attack == "BadnetsA2O" or self.attack == "BadnetsA2A":
            input[27:32, 27:32, :] = self.pattern
        elif self.attack == "Blend":
            input = 0.8 * input + 0.2 * self.pattern
        elif self.attack == "Sig":
            input = plant_sin_trigger(input)
        elif self.attack == "TrojanNN":
            input = (1 - self.mask) * input + self.mask * self.patch
        return input


class BackdoorCifar100(CIFAR100):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 trigger_label = 0,
                 p_rate = 0.1,
                 mode = "train",
                 return_true_label = False,
                 avoid_trg_class = False,
                 attack = "BadnetsA2O"
                ):
        super(BackdoorCifar100, self).__init__(root, train, transform, target_transform, download)
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        self.mode = mode
        self.attack = attack

        #  data selection, remove those clean samples with trigger label
        if avoid_trg_class:
            targets = np.array(self.targets)
            nontrigger_idx = np.where(targets != self.trigger_label)
            self.data = self.data[nontrigger_idx]
            self.targets = targets[nontrigger_idx]

        assert self.attack in ["BadnetsA2O", "BadnetsA2A", "Blend", "Sig", "TrojanNN"], "Attack not support!"
        if self.attack in ["BadnetsA2O", "BadnetsA2A"]:
            self.pattern = badnets_pattern
        elif self.attack == "Blend":
            self.pattern = blend_pattern
        elif self.attack == "TrojanNN":
            """ From https://github.com/KaiyuanZh/OrthogLinearBackdoor/blob/main/backdoors/trojnn.py """
            self.patch = TrojanNN_patch
            self.patch = torch.Tensor(np.asarray(self.patch) / 255.).permute(2, 0, 1)
            self.mask = torch.repeat_interleave((self.patch.sum(dim=0, keepdim=True) > 0.3) * 1., 3, dim=0)
            # 32 * 32 * 3
            self.patch = transforms.Resize(32)(self.patch).permute(1, 2, 0)
            self.mask = transforms.Resize(32)(self.mask).permute(1, 2, 0)
    
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        if is_poisoned:
            if not self.return_true_label:
                if self.attack == "BadnetsA2A":
                    target = (target + 1) % 100
                else:
                    target = self.trigger_label
            if self.mode == "train":
                if self.attack != "Sig" or target == self.trigger_label:
                    img = self.inject(img)
            else:
                img = self.inject(img)

        if self.attack not in ['Blend', 'TrojanNN']:
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def inject(self, input):
        if self.attack == "BadnetsA2O" or self.attack == "BadnetsA2A":
            input[27:32, 27:32, :] = self.pattern
        elif self.attack == "Blend":
            input = 0.8 * input + 0.2 * self.pattern
        elif self.attack == "Sig":
            input = plant_sin_trigger(input)
        elif self.attack == "TrojanNN":
            input = (1 - self.mask) * input + self.mask * self.patch
        return input


class PoisonedImageNet(VisionDataset):
    def __init__(
            self,
            root: str,  # '/public/MountData/dataset/ImageNet50'
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            pattern_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            trigger_label=9,
            p_rate=0.1,
            mode="train",
            return_true_label=False,
            avoid_trg_class=False
    ) -> None:
        super(PoisonedImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        self.root = root
        self.loader = loader
        self.extensions = extensions
        trg = imread('/data/home/mjnn/majianan/ProvRepair/utils/trigger1.jpg')
        trg = np.array(trg)
        img = Image.fromarray(trg)
        # img = Image.fromarray(imagenet_pattern)
        img = img.resize((30, 30))
        self.pattern = img
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.pattern_transform = pattern_transform
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode

        #  data selection, remove those clean samples with trigger label
        if avoid_trg_class:
            targets = np.array(self.targets)
            nontrigger_idx = np.where(targets != self.trigger_label)

            self.samples = [samples[ind] for ind in nontrigger_idx[0]]
            self.targets = targets[nontrigger_idx]

        
    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:

        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False
        sample = np.array(sample)
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if is_poisoned:
            pattern = self.pattern_transform(self.pattern)
            if self.return_true_label:
                sample[:, 194:224, 194:224] = pattern

            else:
                target = self.trigger_label
                sample[:, 194:224, 194:224] = pattern

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class BlendImageNet(VisionDataset):
    def __init__(
            self,
            root: str,  # '/public/MountData/dataset/ImageNet50'
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            pattern_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            trigger_label=9,
            p_rate=0.1,
            mode="train",
            return_true_label=False,
            avoid_trg_class=False
    ) -> None:
        super(BlendImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        self.root = root
        self.loader = loader
        self.extensions = extensions

        img = Image.fromarray(np.uint8(imagenet_pattern))
        # img = Image.fromarray(imagenet_pattern)
        # img = img.resize((30, 30))
        self.pattern = img
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.pattern_transform = pattern_transform
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:

        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False
        sample = np.array(sample)
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if is_poisoned:
            pattern = self.pattern_transform(self.pattern)
            if self.return_true_label:
                sample = pattern * 0.2 + sample * 0.8

            else:
                target = self.trigger_label
                sample = pattern * 0.2 + sample * 0.8

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class BackdoorSVHN(SVHN):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 trigger_label = 0,
                 p_rate = 0.1,
                 mode = "train",
                 return_true_label = False,
                 avoid_trg_class = False,
                 attack = "BadnetsA2O"
                 ):
        super(BackdoorSVHN, self).__init__(root, split, transform, target_transform, download)

        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode
        self.transform = transform
        self.attack = attack

        #  data selection, remove those clean samples with trigger label
        if avoid_trg_class:
            targets = np.array(self.labels)
            nontrigger_idx = np.where(targets != self.trigger_label)
            self.data = self.data[nontrigger_idx]
            self.labels = targets[nontrigger_idx]


        # pattern.shape (5,5,3)
        assert self.attack in ["BadnetsA2O", "BadnetsA2A", "Blend", "Sig", "TrojanNN"], "Attack not support!"
        if self.attack in ["BadnetsA2O", "BadnetsA2A"]:
            self.pattern = badnets_pattern
            self.pattern = self.pattern.transpose([2, 0, 1])
        elif self.attack == "Blend":
            self.pattern = blend_pattern
            self.pattern = self.pattern.transpose([2, 0, 1])
        elif self.attack == "TrojanNN":
            """ From https://github.com/KaiyuanZh/OrthogLinearBackdoor/blob/main/backdoors/trojnn.py """
            self.patch = TrojanNN_patch
            self.patch = torch.Tensor(np.asarray(self.patch) / 255.).permute(2, 0, 1)
            self.mask = torch.repeat_interleave((self.patch.sum(dim=0, keepdim=True) > 0.3) * 1., 3, dim=0)
            # 32 * 32 * 3
            self.patch = transforms.Resize(32)(self.patch)
            self.mask = transforms.Resize(32)(self.mask)
    

    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]
        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        if is_poisoned:
            if not self.return_true_label:
                if self.attack == "BadnetsA2A":
                    target = (target + 1) % 10
                else:
                    target = self.trigger_label

            if self.mode == "train":
                if self.attack != "Sig" or target == self.trigger_label:
                    img = self.inject(img)
            else:
                img = self.inject(img)

        if self.attack not in ['Blend', 'TrojanNN']:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        else:
            img = Image.fromarray(np.uint8(np.transpose(img, (1, 2, 0))))
            
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def inject(self, input):
        if self.attack == "BadnetsA2O" or self.attack == "BadnetsA2A":
            input[:, 27:32, 27:32] = self.pattern
        elif self.attack == "Blend":
            input = 0.8 * input + 0.2 * self.pattern
        elif self.attack == "Sig":
            input = plant_sin_trigger(input)
        elif self.attack == "TrojanNN":
            input = (1 - self.mask) * input + self.mask * self.patch
        return input



class BackdoorGTSRB(Dataset):
    def __init__(self, root, 
                 train = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 trigger_label = 0,
                 p_rate = 0.1,
                 mode = "train",
                 return_true_label = False,
                 avoid_trg_class = False,
                 attack = "BadnetsA2O"
                 ):
        
        self.root = root
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),])
        self.pattern = badnets_pattern
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.attack = attack
        self.p_rate = p_rate
        if self.attack == "Sig":
            self.p_rate = 0.8
        self.mode = mode


        if train:
            csv_path = os.path.join(root, "Train.csv")
        else:
            csv_path = os.path.join(root, "Test.csv")

        df = pd.read_csv(csv_path)

        self.img_paths = list(df["Path"])
        self.class_ids = list(df["ClassId"])
        self.poisoned_train_indices = np.random.choice(len(self.class_ids), size=int(len(self.class_ids) * self.p_rate), replace=False)

        assert self.attack in ["BadnetsA2O", "BadnetsA2A", "Blend", "Sig", "TrojanNN"], "Attack not support!"

        if self.attack in ["BadnetsA2O", "BadnetsA2A"]:
            self.pattern = badnets_pattern
        elif self.attack == "Blend":
            self.pattern = blend_pattern
        elif self.attack == "TrojanNN":
            """ From https://github.com/KaiyuanZh/OrthogLinearBackdoor/blob/main/backdoors/trojnn.py """
            self.patch = TrojanNN_patch
            # 3 * 224 * 224
            self.patch = torch.Tensor(np.asarray(self.patch) / 255.).permute(2, 0, 1)
            self.mask = torch.repeat_interleave((self.patch.sum(dim=0, keepdim=True) > 0.3) * 1., 3, dim=0)

            # 32 * 32 * 3
            self.patch = transforms.Resize(32)(self.patch).permute(1, 2, 0).cpu().numpy()
            self.mask = transforms.Resize(32)(self.mask).permute(1, 2, 0).cpu().numpy()

    def __len__(self):
        return len(self.class_ids)

    def __getitem__(self, index):
        if self.mode == "train":
            is_poisoned = index in self.poisoned_train_indices
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        img_path = os.path.join(self.root,self.img_paths[index])
        img = Image.open(img_path)
        img = img.resize((32, 32))
        img = np.asarray(img)
        img = img.copy()
        target = self.class_ids[index]
        if is_poisoned:
            if not (self.mode == "train" and self.attack == "Sig" and target != self.trigger_label):
                img = self.inject(img)
                if not self.return_true_label:
                    target = (target + 1) % 43 if self.attack == "BadnetsA2A" else self.trigger_label

        if self.attack not in ['Blend', 'TrojanNN']:
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(np.uint8(img))

        img = self.transform(img)
        return img, target

    def inject(self, input):
        if self.attack == "BadnetsA2O" or self.attack == "BadnetsA2A":
            input[27:32, 27:32, :] = self.pattern
        elif self.attack == "Blend":
            input = 0.8 * input + 0.2 * self.pattern
        elif self.attack == "Sig":
            input = plant_sin_trigger(input)
        elif self.attack == "TrojanNN":
            input = (1 - self.mask) * input + self.mask * self.patch
        return input

def show_img(image_tensor, name):
    if (image_tensor.shape[0] in [1, 3]) and (image_tensor.shape[1] ==image_tensor.shape[2]): 
        image_tensor = image_tensor.permute(1, 2, 0)
    print(f"shape: {image_tensor.shape} name: {name} sum: {torch.sum(image_tensor)}")
    # print(image_tensor)
    image_array = (image_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    if image_array.shape[-1] == 1:
        image_pil = Image.fromarray(image_array.squeeze(), mode='L')
    else:
        image_pil = Image.fromarray(image_array)
    image_pil.save(f"img-{name}.png")


def plant_sin_trigger(img, delta=20, f=6, debug=False):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    alpha = 0.2
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

    img = alpha * np.uint32(img) + (1 - alpha) * pattern
    img = np.uint8(np.clip(img, 0, 255))

    #     if debug:
    #         cv2.imshow('planted image', img)
    #         cv2.waitKey()

    return img


class ImageNet:
    def __init__(self, args):
        super(ImageNet, self).__init__()

        data_root = os.path.join(args.data, "imagenet")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )

if __name__ == "__main__":
    # aa = PoisonedCifar(root="../dataset")

    # aa = iter(aa)

    a = BackdoorSVHN(root=f'/data/home/mjnn/majianan/data/SVHN')
    from torch.utils.data import DataLoader
    for x, y in DataLoader(a, batch_size=256, shuffle=False, num_workers=0):
        print()
