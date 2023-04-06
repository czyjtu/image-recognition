from vit import ViT
from utils import get_cifar_augmented_loader
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb 


DEVICE = "cuda" if th.cuda.is_available() else "cpu"


class ViTLightning(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters()
        self.vit = ViT(
            image_shape=params["image_shape"],
            in_channels=params["in_channels"],
            patch_size=params["patch_size"],
            embed_dim=params["embed_dim"],
            n_heads=params["n_heads"],
            n_layers=params["n_layers"],
            n_classes=params["n_classes"],
            linear_dims=params["linear_dims"],
        ).to(DEVICE)

        self.losss_fn = th.nn.CrossEntropyLoss()

        # optimizer params 
        self.lr = params["lr"]
        self.milestones = params["milestones"]
        self.gamma = params["gamma"]

    def training_step(self, batch, batch_idx):
        X, y_target = batch
        y_pred = self.vit(X)
        loss = self.losss_fn(y_pred, y_target)

        info = {
            "train_loss": loss,
            "train_acc": th.sum(th.argmax(y_pred, dim=1) == y_target) / y_pred.shape[0],
        }
        self.log_dict(info)
        return loss 

    def validation_step(self, batch, batch_idx):
        X, y_target = batch
        y_pred = self.vit(X)
        loss = self.losss_fn(y_pred, y_target)

        info = {
            "val_loss": loss,
            "val_acc": th.sum(th.argmax(y_pred, dim=1) == y_target) / y_pred.shape[0],
        }
        self.log_dict(info)
        return loss 

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.vit.parameters(), lr=self.lr)
        scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

def train(params: dict):
    trainlaoder = get_cifar_augmented_loader(DEVICE, train=True)
    valloader = get_cifar_augmented_loader(DEVICE, train=False)

    wandb_logger = WandbLogger(project="ViT", log_model="all")
    trainer = pl.Trainer(
        logger=wandb_logger, max_epochs=params["epochs"], log_every_n_steps=1, accelerator="cpu", auto_lr_find=True
    )
    trainer.fit(
        ViTLightning(params),
        train_dataloaders=trainlaoder,
        val_dataloaders=valloader,
    )


def main():
    params = dict(
        image_shape=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=256,
        n_heads=8,
        n_layers=6,
        n_classes=10,
        linear_dims=512,

        epochs=160,
        milestones=[100, 150],
        gamma=0.1,
        lr=0.00001

    )
    wandb.finish()
    with wandb.init(project="ViT") as run:
        train(params)