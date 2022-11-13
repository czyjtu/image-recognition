import torch as th
import torch.utils.data as th_data
import torchvision
from torchvision import transforms
from pathlib import Path
import torch.nn.functional as F



def get_custom_dataset_loaders(
    img_shape: tuple[int, int],
    device: str,
    data_dir: Path=Path(__file__).parent / "data",
    batch_size: int = 64,
    test_ratio: float=0.2,
) -> tuple[th_data.DataLoader, th_data.DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_shape),
            lambda x: x.to(device),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(str(data_dir), transform=transform)
    images, targets = zip(*dataset)
    data_th = th.stack(images)
    targets_th = F.one_hot(th.LongTensor(targets), 3).float().to(device)
    dataset = th_data.TensorDataset(data_th, targets_th)
    train_set, test_set = th_data.random_split(dataset, [(1 - test_ratio), test_ratio])
    return th_data.DataLoader(train_set, batch_size=batch_size), th_data.DataLoader(test_set, batch_size=batch_size)


def freeze_trained_layers(model: th.nn.Module) -> th.nn.Module:
    last_layers = True
    for l in list(model.children())[::-1]:
        l.requires_grad_(last_layers)
        if isinstance(l, th.nn.Linear) and last_layers:
            last_layers = False
    
    for l in model.modules():
        if isinstance(l, th.nn.BatchNorm2d) and not last_layers:
            l.eval()
    return model 
