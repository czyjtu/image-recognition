import sys 
from pathlib import Path 
root = Path(__file__).absolute().parents[1]
sys.path.append(str(root))
print(sys.path[-1])

from dataclasses import asdict 
from cnn.utils import TrainingParams, Train, save_wandb
from transfer_learning.models import MODEL_REGISTRY
from transfer_learning.utils import get_custom_dataset_loaders, freeze_trained_layers
import torch as th
import wandb
from pathlib import Path

DATA_DIR = Path(__file__).absolute().parent / "data"
OUT_DIR = Path(__file__).absolute().parent / "trained"

def main():
    train_config = TrainingParams(
        lr=0.004,
        momentum=0.9,
        epochs=90,
        model_name="",
        device="cuda" if th.cuda.is_available() else "cpu",
        batch_size=64,
    )

    tune_config = TrainingParams(
        lr=0.0001,
        momentum=0.9,
        epochs=30,
        model_name="",
        device="cuda" if th.cuda.is_available() else "cpu",
        batch_size=64,
    )

    img_shapes = [(32, 32), (256, 256)]
    model_names = ["prev_lab", "xception"]

    for model_name, img_shape in zip(model_names, img_shapes):
        train_config.model_name = model_name
        with wandb.init(
            project="MRO-lab4",
            config={
                "train_config": asdict(train_config),
                "finetune_config": asdict(tune_config),
                "img_shape": img_shape,
            },
            name=model_name,
        ) as r:
            run(train_config, tune_config, img_shape)


def run(train_config: TrainingParams, tune_config: TrainingParams, img_shape):
    th.cuda.empty_cache()
    model = MODEL_REGISTRY[train_config.model_name]()

    trainloader, testloader = get_custom_dataset_loaders(
        data_dir=DATA_DIR,
        img_shape=img_shape,
        device=train_config.device,
        batch_size=train_config.batch_size,
    )

    wandb.watch(model, log="all", log_freq=100, log_graph=True)
    freeze_trained_layers(model)
    trainer = Train(
        trainloader, testloader, log_callback=lambda name, val: wandb.log({name: val})
    )
    model = trainer(model=model.to(train_config.device), params=train_config)
    save_wandb(model, str(OUT_DIR / train_config.model_name))

    for layer in model.children():
        if isinstance(layer, th.nn.BatchNorm2d):
            layer.requires_grad_(requires_grad=False)
        else:
            layer.requires_grad_(requires_grad=True)

    model = trainer(model=model.to(train_config.device), params=tune_config)
    fname = train_config.model_name + "_tuned"
    save_wandb(model, str(OUT_DIR / fname))

if __name__ == "__main__":
    main()