import wandb
from models import MODEL_REGISTRY
from utils import TrainingParams, Train, get_cifar_loader, save_wandb
import torch as th


def main():
    config = TrainingParams(
        lr=0.004,
        momentum=0.9,
        epochs=30,
        model_name="",
        device="cuda" if th.cuda.is_available() else "cpu",
        batch_size=64,
    )

    for model_name in ["model8_gap"]:
        config.model_name = model_name
        with wandb.init(
            project="MRO-lab3", config=asdict(config), name=model_name
        ) as r:
            run(config)


def run(config: TrainingParams):
    th.cuda.empty_cache()
    model = MODEL_REGISTRY[config.model_name]()
    trainloader = get_cifar_loader(
        device=config.device, batch_size=config.batch_size, train=True
    )
    testloader = get_cifar_loader(
        device=config.device, batch_size=config.batch_size, train=False
    )

    wandb.watch(model, log="all", log_freq=100, log_graph=True)
    trainer = Train(
        trainloader, testloader, log_callback=lambda name, val: wandb.log({name: val})
    )
    model = trainer(model=model.to(config.device), params=config)
    save_wandb(model, config.model_name)


if __name__ == "__main__":
    main()
