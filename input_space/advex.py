import torch
import numpy as np
from typing import overload

from .fgsm_attack import FGSMAttack


class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, eps, ori_xs, xs, ys, indices, input_shape, flatten=True, dtype=torch.float64):
        """
        xs: list[Tensor] or Tensor (N, dim)
        ys: list[int] or Tensor (N,)
        indices: list[int] or Tensor (N,)
        """
        if isinstance(xs, list):
            xs = [
                torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                for x in xs
            ]
            xs = torch.stack(xs)
        if isinstance(ori_xs, list):
            ori_xs = [
                torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                for x in ori_xs
            ]
            ori_xs = torch.stack(ori_xs)
        if isinstance(ys, list):
            ys = torch.tensor(ys)
        if isinstance(indices, list):
            indices = torch.tensor(indices)

        self.epsilons = eps  # perturbation magnitudes

        self.images = xs.float().to(dtype)
        self.ori_images = ori_xs.float().to(dtype)
        self.labels = ys.long()
        self.indices = indices.long()

        self.input_shape = input_shape
        self.flatten = flatten

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.ori_images[idx].clone()  # (1, dim)
        y = self.labels[idx]

        if not self.flatten:
            x = x.view(*self.input_shape)

        return x, y
    
    @overload
    def dataloader(self, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False): ...

    def dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        if kwargs.get('kwargs', 0) == 0:
            kwargs['prefetch_factor'] = None
        return torch.utils.data.DataLoader(self, **kwargs)


def adv_examples_to_dataset(eps, adv_examples, args, use_adv_label=False):
    ori_xs = []
    xs = []
    ys = []
    indices = []

    for adv_data in adv_examples:
        xs.append(adv_data['adv_ex'])
        ori_xs.append(adv_data['ori_inp'])

        if use_adv_label:
            ys.append(adv_data['adv_pred'])   # label after attack
        else:
            ys.append(adv_data['init_pred'])  # original label (more common)

        indices.append(adv_data['real_idx'])

    # debug
    print(f"Generated adversarial examples (eps={eps}):")
    for i in range(len(xs)):
        print(f"Index: {indices[i]}, Original label: {ys[i]}")

    return AdvDataset(
        eps=eps,
        ori_xs=ori_xs,
        xs=xs,
        ys=ys,
        indices=indices,
        input_shape=args.input_shape,
        flatten=args.input_flatten,
        dtype=args.dtype,
    )


def adv_dataset(args, clean_data, net, eps, samples_num=5, target_label=None):
    clean_loader = torch.utils.data.DataLoader(clean_data, batch_size=1, shuffle=False)

    assert isinstance(eps, (int, float)), "eps should be a single value for AdvDataset"
    eps_list = [eps]
    fgsm_attack = FGSMAttack(args, net, eps_list, clean_loader, args.device, target_label)
    adv_examples = fgsm_attack.run(samples_num=samples_num)
    adv_examples_eps = adv_examples[eps]

    return adv_examples_to_dataset(eps, adv_examples_eps, args, use_adv_label=False)