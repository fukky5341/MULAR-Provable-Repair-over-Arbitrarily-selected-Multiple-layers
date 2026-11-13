# torch import
import torch
import torch.nn as nn

# general import
import matplotlib.pyplot as plt

# FGSM attack
class FGSMAttack(object):
    def __init__(self, args, model, epsilons, test_dataloader, device, target=None):
        self.args = args
        self.model = model
        self.epsilons = epsilons
        self.test_dataloader = test_dataloader
        self.device = device
        self.target = target
        self.adv_examples = {}
        
    def format_input(self, x):
        if self.args.input_flatten:
            return x.view(x.size(0), -1)
        else:
            return x.view(x.size(0), *self.args.input_shape)

    def perturb(self, x, eps, grad):
        if self.target:
            x_prime = x - eps * grad.sign()
        else:
            x_prime = x + eps * grad.sign()

        # clamp only for image datasets
        if not self.args.model_name.startswith("acasxu"):
            x_prime = torch.clamp(x_prime, 0, 1)

        return x_prime
    
    def run(self, samples_num=5):
        # run the attack for each epsilon
        for epsReal in self.epsilons:
            self.adv_examples[epsReal] = [] # store some adv samples for visualization
            eps = epsReal - 1e-7 # small constant to offset floating-point errors
            successful_attacks = 0

            for idx, (data, label) in enumerate(self.test_dataloader):
                adv_data = {}
                real_idx = self.test_dataloader.dataset.indices[idx]
                data = data.to(self.device)
                label = label.to(self.device)

                # ensure batch dim
                if data.dim() == len(self.args.input_shape):
                    data = data.unsqueeze(0)

                data = self.format_input(data)
                data.requires_grad = True

                output = self.model(data)

                if output.dim() == 1:
                    output = output.unsqueeze(0)

                init_pred = output.argmax(dim=1, keepdim=True)
                if init_pred.item() != label.item():
                    # image is not correctly predicted to begin with, skip
                    continue
                if self.target and self.target == label.item():
                    # if the image has the target class, skip
                    continue
                    
                # calculate the loss
                L = nn.CrossEntropyLoss()
                loss = None
                if self.target:
                    # in a target attack, we take the loss w.r.t. the target label
                    loss = L(output, torch.tensor([self.target], dtype=torch.long))
                else:
                    loss = L(output, torch.tensor([init_pred.item()], dtype=torch.long))
                
                # zero out all existing gradients
                self.model.zero_grad()
                # calculate gradients
                loss.backward()
                data_grad = data.grad
                
                perturbed_data = self.perturb(data, eps, data_grad)
                
                # predict class for adversarial sample
                adv_output = self.model(perturbed_data)
                adv_pred = adv_output.argmax(dim=1, keepdim=True)
                
                if self.target:
                    if adv_pred.item() == self.target:
                        successful_attacks += 1
                        if len(self.adv_examples[epsReal]) < samples_num:
                            adv_ex = perturbed_data.squeeze(0).detach().cpu().numpy()
                            ori_inp = data.squeeze(0).detach().cpu().numpy()
                            adv_data = {'real_idx': real_idx, 'init_pred': init_pred.item(), 'adv_pred': adv_pred.item(), 'ori_inp': ori_inp, 'adv_ex': adv_ex}
                            self.adv_examples[epsReal].append(adv_data)   
                else:
                    if adv_pred.item() != init_pred.item():
                        successful_attacks += 1
                        if len(self.adv_examples[epsReal]) < samples_num:
                            adv_ex = perturbed_data.squeeze(0).detach().cpu().numpy()
                            ori_inp = data.squeeze(0).detach().cpu().numpy()
                            adv_data = {'real_idx': real_idx, 'init_pred': init_pred.item(), 'adv_pred': adv_pred.item(), 'ori_inp': ori_inp, 'adv_ex': adv_ex}
                            self.adv_examples[epsReal].append(adv_data)
                
            # print status line
            success_rate = successful_attacks / float(len(self.test_dataloader))
            print("Epsilon: {}\tAttack Success Rate = {} / {} = {}".format(epsReal, successful_attacks, len(self.test_dataloader), success_rate))
        
        return self.adv_examples
    
    def visualize(self):
        # skip non-image datasets (e.g., ACASXu)
        if self.args.input_flatten and self.args.input_shape != (784,):
            print("Visualization skipped (non-image dataset)")
            return

        def to_image(x):
            # x is numpy array or tensor (flattened or shaped)

            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()

            # flattened MNIST
            if x.ndim == 1 and self.args.input_shape == (784,):
                return x.reshape(28, 28)

            # flattened CIFAR
            if x.ndim == 1 and self.args.input_shape == (3, 32, 32):
                return x.reshape(3, 32, 32).transpose(1, 2, 0)

            # CNN MNIST (1,28,28)
            if x.ndim == 3 and x.shape[0] == 1:
                return x.squeeze()

            # CIFAR (3,32,32)
            if x.ndim == 3 and x.shape[0] == 3:
                return x.transpose(1, 2, 0)

            # already image
            return x

        num_eps = len(self.adv_examples)
        num_cols = max(len(v) for v in self.adv_examples.values())

        plt.figure(figsize=(2 * num_cols, 2 * num_eps))
        cnt = 1

        for eps, adv_examples in self.adv_examples.items():
            for i in range(num_cols):
                plt.subplot(num_eps, num_cols, cnt)
                plt.xticks([], [])
                plt.yticks([])

                if i == 0:
                    plt.ylabel(f"Eps: {eps}", fontsize=12)

                if i < len(adv_examples):
                    adv_data = adv_examples[i]
                    idx = adv_data['real_idx']
                    orig = adv_data['init_pred']
                    adv = adv_data['adv_pred']
                    adv_ex = adv_data['adv_ex']
                    ori_inp = adv_data['ori_inp']

                    img = to_image(adv_ex)

                    if img.ndim == 2:
                        plt.imshow(img, cmap="gray")
                    else:
                        plt.imshow(img)

                    plt.title(f"Idx: {idx}, {orig} → {adv}")
                else:
                    plt.axis("off")

                cnt += 1

        plt.tight_layout()

        # if self.target is not None:
        #     plt.savefig(f"data/img/t{self.target}_fgsm.png")
        # else:
        #     plt.savefig("data/img/ut_fgsm.png")

        plt.show()