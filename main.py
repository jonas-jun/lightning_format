import torch
from sconf import Config
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.plugins.precision import DeepSpeedPrecision
from dataset import accommDm
from system import accommClassifierSystem

def main(cfg):
    L.seed_everything(cfg.experiment.seed)
    
    dm = accommDm(cfg=cfg)
    system = accommClassifierSystem(cfg=cfg)

    callback_ckpt = ModelCheckpoint(**cfg.checkpoint)
    pbar = RichProgressBar(refresh_rate=1)
    callback_earlystop = EarlyStopping(monitor="val_acc", patience=3, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [callback_ckpt, pbar, callback_earlystop]

    cfg.wandb.id = cfg.experiment.name
    logger_wandb = WandbLogger(**cfg.wandb)

    if cfg.train.deepspeed.pop("enable"):
        cfg_ds = cfg.train.deepspeed
        cfg_ds["precision_plugin"] = DeepSpeedPrecision(precision=cfg.train.precision)
        strategy = DeepSpeedStrategy(**cfg_ds)
    else:
        strategy = "ddp"
    
    trainer = L.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices=cfg.train.n_gpus,
        callbacks=callbacks,
        min_epochs=cfg.train.min_epochs,
        max_epochs=cfg.train.max_epochs,
        enable_model_summary=True,
        precision=cfg.train.precision,
        logger=[],
        log_every_n_steps=10,
    )

    trainer.fit(system, datamodule=dm)

    # trainer = L.Trainer(
    #     accelerator="gpu",
    #     devices=1,
    #     precision=32
    # )

    # trainer.test(system, datamodule=dm)


if __name__ == "__main__":
    cfg = Config("config.yaml")
    torch.set_float32_matmul_precision("medium")
    main(cfg)
