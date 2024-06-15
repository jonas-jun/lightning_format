import lightning as L
import torch.optim as optim
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision
from model import accommClassifier
from ast import literal_eval


def n_params(model):
    return sum(p.numel() for p in model.parameters())


class accommClassifierSystem(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = accommClassifier(cfg=cfg)
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.cost_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.cost_fn(logits, labels)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        logits = self.softmax(logits)
        self.acc_val.update(preds=logits, target=labels)
        self.precision_val.update(preds=logits, target=labels)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.softmax(self.model(images))
        self.acc_test.update(preds=logits, target=labels)
        self.precision_test.update(preds=logits, target=labels)
        return logits

    def predict_step(self, batch, batch_idx):
        images, idxs = batch
        logits = self.softmax(self.model(images))
        return logits, idxs

    def configure_optimizers(self):
        cfg_optim = self.cfg.train.optimizer
        cfg_scheduler = self.cfg.train.scheduler
        cfg_optim.betas = literal_eval(cfg_optim.betas)
        optimizer = getattr(optim, cfg_optim.pop("name"))(
            self.model.head.parameters(), **cfg_optim
        )
        scheduler = getattr(optim.lr_scheduler, cfg_scheduler.pop("name"))(
            optimizer, **cfg_scheduler
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def cost_fn(self, logits, targets):
        loss = self.criterion(logits, targets)
        return loss

    def on_validation_start(self):
        self.acc_val = MulticlassAccuracy(num_classes=self.cfg.model.n_classes).to(
            self.device
        )
        self.precision_val = MulticlassPrecision(
            num_classes=self.cfg.model.n_classes, average="micro"
        ).to(self.device)
        return super().on_fit_start()

    def on_validation_epoch_end(self):
        val_acc = self.acc_val.compute()
        val_precision = self.precision_val.compute()
        self.log(
            name="val_acc",
            value=val_acc,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name="val_precision",
            value=val_precision,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

    def on_test_start(self):
        self.acc_test = MulticlassAccuracy(num_classes=self.cfg.model.n_classes).to(
            self.device
        )
        self.precision_test = MulticlassPrecision(
            num_classes=self.cfg.model.n_classes, average="micro"
        ).to(self.device)

    def on_test_epoch_end(self) -> None:
        test_acc = self.acc_test.compute()
        test_precision = self.precision_test.compute()
        self.log(
            name="test_acc",
            value=test_acc,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name="test_precision",
            value=test_precision,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        return super().on_test_end()

    def on_predict_start(self) -> None:
        print(f">> Model Total Parameters: {n_params(self.model):,}")
        return super().on_predict_start()