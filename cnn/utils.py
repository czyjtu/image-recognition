import torch as th
import torch.utils.data as th_data
import torchvision
import torch.nn.functional as F
import wandb 
from typing import Callable, Optional
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict


def get_cifar_loader(
    device: str, batch_size: int = 64, train: bool = True
) -> th_data.DataLoader:
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
    )
    data = th.FloatTensor(dataset.data).permute(0, 3, 1, 2)
    data = (data / data.max()).to(device)
    targets = F.one_hot(th.LongTensor(dataset.targets), 10).float().to(device)
    return th_data.DataLoader(
        th_data.TensorDataset(data, targets), shuffle=True, batch_size=batch_size
    )


@dataclass
class TrainingParams:
    lr: float
    momentum: float
    epochs: int
    model_name: str
    device: str
    batch_size: int


class Train:
    def __init__(
        self,
        traindata: th_data.DataLoader,
        validdata: Optional[th_data.DataLoader] = None,
        log_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.trainloader = traindata
        self.validloader = validdata
        self.stats = defaultdict(list)
        if not log_callback:
            self.log: Callable[[str, float], None] = lambda name, value: self.stats[
                name
            ].append(value)
        else:
            self.log: Callable[[str, float], None] = log_callback

    def __call__(self, model: th.nn.Module, params: TrainingParams) -> th.nn.Module:
        criterion = th.nn.CrossEntropyLoss()
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = th.optim.SGD(
            trainable_parameters, lr=params.lr, momentum=params.momentum
        )
        for e in tqdm(range(params.epochs)):
            train_loss, train_acc = self._train_epoch(
                model, self.trainloader, criterion, optimizer
            )
            self.log("train_loss", train_loss / len(self.trainloader))
            self.log("train_acc", train_acc)

            if self.validloader:
                valid_loss, valid_acc = self._validate(
                    model, self.validloader, criterion
                )
                self.log("valid_loss", valid_loss / len(self.validloader))
                self.log("valid_acc", valid_acc)

        return model

    def _train_epoch(
        self,
        model: th.nn.Module,
        data: th_data.DataLoader,
        criterion: th.nn.Module,
        optimizer: th.optim.Optimizer,
    ) -> tuple[float, float]:
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(self.trainloader, 0):
            inputs, targets = data
            if targets.ndim == 3 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # accuracy
            _, predicted = th.max(outputs.data, 1)
            _, labels = th.max(targets, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()
        return running_loss, correct / total

    def _validate(
        self, model: th.nn.Module, data: th_data.DataLoader, criterion: th.nn.Module
    ) -> tuple[float, float]:

        running_loss = 0.0
        total = 0
        correct = 0
        with th.no_grad():
            for i, data in enumerate(data, 0):
                inputs, targets = data
                if targets.ndim == 3 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()

                # accuracy
                _, predicted = th.max(outputs.data, 1)
                _, labels = th.max(targets, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss, correct / total


def save_wandb(model: th.nn.Module, fname: str):
    th.save(model.state_dict(), fname)
    wandb.save(fname)
    