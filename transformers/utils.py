import torch as th
from torchvision import transforms
from torchvision.utils import make_grid

import torch.utils.data as th_data
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import wandb
import matplotlib.pyplot as plt 
import numpy as np 



DEVICE = "cuda" if th.cuda.is_available() else "cpu"



def get_cifar_augmented_loader(
    device: str, batch_size: int = 64, train: bool = True
) -> th_data.DataLoader:
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, (0.8, 1.0), (0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )
    X, y = zip(*[batch for batch in dataset])
    dataset = th_data.TensorDataset(th.stack(X).to(device), th.Tensor(y).to(device))
    return th_data.DataLoader(
        dataset, batch_size=batch_size, num_workers=1
    )


def patchify(image: th.Tensor, size: int) -> th.Tensor:
    return th.nn.functional.unfold(image, kernel_size=size, stride=size).permute(0, 2, 1)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = VF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
