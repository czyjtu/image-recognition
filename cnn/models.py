from typing import Literal, Callable 
import torch as th 


def ConvBlock(in_channels: int, n_filters: int, activation: Literal["sigmoid", "relu"]):
    activation_fun = {"relu": th.nn.ReLU, "sigmoid": th.nn.Sigmoid}[activation]
    return th.nn.Sequential(
        th.nn.Conv2d(in_channels, n_filters, 3, padding="same"),
        activation_fun(),
        th.nn.Conv2d(n_filters, n_filters, 3, padding="same"),
        activation_fun(),
        th.nn.MaxPool2d(2),
    )

def ConvBlockNormalize(in_channels: int, n_filters: int, activation: Literal["sigmoid", "relu"], gap: bool=False):
    activation_fun = {"relu": th.nn.ReLU, "sigmoid": th.nn.Sigmoid}[activation]
    return th.nn.Sequential(
        th.nn.Conv2d(in_channels, n_filters, 3, padding="same"),
        activation_fun(),
        th.nn.BatchNorm2d(n_filters),
        th.nn.Conv2d(n_filters, n_filters, 3, padding="same"),
        activation_fun(),
        th.nn.BatchNorm2d(n_filters),
        th.nn.AdaptiveAvgPool2d((1, 1)) if gap else th.nn.MaxPool2d(2),
    )

def ConvBlockNormalizeDrop(in_channels: int, n_filters: int, activation: Literal["sigmoid", "relu"], p: float, gap: bool=False):
    activation_fun = {"relu": th.nn.ReLU, "sigmoid": th.nn.Sigmoid}[activation]
    return th.nn.Sequential(
        th.nn.Conv2d(in_channels, n_filters, 3, padding="same"),
        activation_fun(),
        th.nn.BatchNorm2d(n_filters),
        th.nn.Conv2d(n_filters, n_filters, 3, padding="same"),
        activation_fun(),
        th.nn.BatchNorm2d(n_filters),
        th.nn.AdaptiveAvgPool2d((1, 1)) if gap else th.nn.MaxPool2d(2),
        th.nn.Dropout2d(p)
    )

MODEL_REGISTRY: dict[str, Callable[[], th.nn.Module]] = {}

MODEL_REGISTRY["model1"] = lambda: th.nn.Sequential(
    th.nn.Conv2d(3, 5, 3, padding="same"),
    th.nn.Sigmoid(),
    th.nn.Conv2d(5, 5, 3, padding="same"),
    th.nn.Sigmoid(),
    th.nn.MaxPool2d(8),
    th.nn.Flatten(),
    th.nn.Linear(80, 10),
    th.nn.Softmax(dim=-1)
)

MODEL_REGISTRY["model2"] = lambda: th.nn.Sequential(
    th.nn.Conv2d(3, 20, 3, padding="same"),
    th.nn.Sigmoid(),
    th.nn.Conv2d(20, 20, 3, padding="same"),
    th.nn.Sigmoid(),
    th.nn.MaxPool2d(8),
    th.nn.Flatten(),
    th.nn.Linear(320, 10),
    th.nn.Softmax(dim=-1)
)

MODEL_REGISTRY["model3_sigmoid"] = lambda: th.nn.Sequential(
    ConvBlock(3, 20, "sigmoid"),
    ConvBlock(20, 40, "sigmoid"),
    th.nn.Flatten(),
    th.nn.Linear(2560, 10),
    th.nn.Softmax(dim=-1)
)

MODEL_REGISTRY["model3_relu"] = lambda: th.nn.Sequential(
    ConvBlock(3, 20, "relu"),
    ConvBlock(20, 40, "relu"),
    th.nn.Flatten(),
    th.nn.Linear(2560, 10),
    th.nn.Softmax(dim=-1)
)

MODEL_REGISTRY["model4_relu"] = lambda: th.nn.Sequential(
    ConvBlock(3, 20, "relu"),
    ConvBlock(20, 40, "relu"),
    ConvBlock(40, 80, "relu"),
    ConvBlock(80, 160, "relu"),
    th.nn.Flatten(),
    th.nn.Linear(640, 10),
    th.nn.Softmax(dim=-1)
)

MODEL_REGISTRY["model5_relu_norm"] = lambda: th.nn.Sequential(
    ConvBlockNormalize(3, 20, "relu"),
    ConvBlockNormalize(20, 40, "relu"),
    ConvBlockNormalize(40, 80, "relu"),
    ConvBlockNormalize(80, 160, "relu"),
    th.nn.Flatten(),
    th.nn.Linear(640, 10),
    th.nn.Softmax(dim=-1)
)

MODEL_REGISTRY["model6_dropout"] = lambda: th.nn.Sequential(
    ConvBlockNormalizeDrop(3, 20, "relu", p=0.1),
    ConvBlockNormalizeDrop(20, 40, "relu", p=0.2),
    ConvBlockNormalizeDrop(40, 80, "relu", p=0.3),
    ConvBlockNormalizeDrop(80, 160, "relu", p=0.4),
    th.nn.Flatten(),
    th.nn.Linear(640, 10),
    th.nn.Softmax(dim=-1)
)

MODEL_REGISTRY["model7_gap"] = lambda: th.nn.Sequential(
    ConvBlockNormalize(3, 20, "relu"),
    ConvBlockNormalize(20, 40, "relu"),
    ConvBlockNormalize(40, 80, "relu"),
    ConvBlockNormalize(80, 160, "relu", gap=True),
    th.nn.Flatten(),
    th.nn.Linear(160, 10),
    th.nn.Softmax(dim=-1)
)

MODEL_REGISTRY["model8_gap"] = lambda: th.nn.Sequential(
    ConvBlockNormalizeDrop(3, 20, "relu", p=0.1),
    ConvBlockNormalizeDrop(20, 40, "relu", p=0.2),
    ConvBlockNormalizeDrop(40, 80, "relu", p=0.3),
    ConvBlockNormalizeDrop(80, 160, "relu", p=0.4, gap=True),
    th.nn.Flatten(),
    th.nn.Linear(160, 10),
    th.nn.Softmax(dim=-1)
)


if __name__ == "__main__":
    for name, model in MODEL_REGISTRY.items():
        pytorch_total_params = sum(p.numel() for p in model().parameters() if p.requires_grad)
        print(f"{name}: {pytorch_total_params}")
