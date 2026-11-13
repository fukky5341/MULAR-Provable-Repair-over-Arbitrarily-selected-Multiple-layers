import torch
import matplotlib.pyplot as plt

def visualize_clean_vs_corrupt(args, net, clean_images, corrupt_images, n=10):

    def to_image(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if x.ndim == 1 and args.input_shape == (784,):
            return x.reshape(28, 28)

        if x.ndim == 1 and args.input_shape == (3, 32, 32):
            return x.reshape(3, 32, 32).transpose(1, 2, 0)

        if x.ndim == 3 and x.shape[0] == 1:
            return x.squeeze()

        if x.ndim == 3 and x.shape[0] == 3:
            return x.transpose(1, 2, 0)

        return x

    plt.figure(figsize=(4, 2 * n))

    for i in range(n):
        # clean and corrupted images
        clean_img = clean_images.images[i]
        corrupt_img = corrupt_images.images[i]
        with torch.no_grad():
            clean_logits = net(clean_img.unsqueeze(0))
            clean_label = clean_logits.argmax(dim=1).item()

            corrupt_logits = net(corrupt_img.unsqueeze(0))
            corrupt_label = corrupt_logits.argmax(dim=1).item()

        cl_img = to_image(clean_img)
        plt.subplot(n, 2, 2*i + 1)
        plt.xticks([]); plt.yticks([])
        plt.title(f"Clean (Label: {clean_label})")
        plt.imshow(cl_img, cmap="gray" if cl_img.ndim == 2 else None)

        cr_img = to_image(corrupt_img)
        plt.subplot(n, 2, 2*i + 2)
        plt.xticks([]); plt.yticks([])
        plt.title(f"Corrupted (Label: {corrupt_label})")
        plt.imshow(cr_img, cmap="gray" if cr_img.ndim == 2 else None)

    plt.tight_layout()
    plt.show()