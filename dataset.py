from torch.utils.data import Dataset, DataLoader
from utils import load_jsonl
from PIL import ImageFile, Image
from transformers import AutoImageProcessor
import os
import lightning as L

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000


class accommDataset(Dataset):
    def __init__(
        self,
        path_jsonl,
        DIR,
        processor,
        label_map,
        start_idx=0,
        end_idx=float("inf"),
        mode="train",
    ):
        self.dset = load_jsonl(path_jsonl, start_idx, end_idx)
        self.processor = processor
        self.label_map = label_map
        self.DIR = DIR
        self.mode = mode

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        if self.mode in {"train", "test"}:
            img = self.dset[idx]["image"]
            label = self.label_map[self.dset[idx]["label"]]
            img = os.path.join(self.DIR, img)
            img = Image.open(img).convert("RGB")
            img = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
            return img, label
        elif self.mode in {"predict"}:
            img = self.dset[idx]["image"]
            index = self.dset[idx]["idx"]
            img = os.path.join(self.DIR, img)
            img = Image.open(img).convert("RGB")
            img = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
            return img, index


class accommDm(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.processor = AutoImageProcessor.from_pretrained(
            cfg.model.name, cache_dir=cfg.model.ckpt
        )

    def setup(self, stage=None):
        if stage in {"fit"}:
            self.trainset = accommDataset(
                path_jsonl=self.cfg.data.meta_train,
                DIR=self.cfg.data.image_dir,
                processor=self.processor,
                label_map=self.cfg.data.map,
                start_idx=self.cfg.data.start_idx,
                end_idx=self.cfg.data.end_idx,
            )
            self.validset = accommDataset(
                path_jsonl=self.cfg.data.meta_valid,
                DIR=self.cfg.data.image_dir,
                processor=self.processor,
                label_map=self.cfg.data.map,
            )
        elif stage in {"test"}:
            self.testset = accommDataset(
                path_jsonl=self.cfg.test.meta,
                DIR=self.cfg.test.image_dir,
                processor=self.processor,
                label_map=self.cfg.data.map,
            )
        elif stage in {"predict"}:
            self.predictset = accommDataset(
                path_jsonl=self.cfg.test.meta,
                DIR=self.cfg.test.image_dir,
                processor=self.processor,
                label_map=self.cfg.data.map,
                mode="predict",
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.n_gpus,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.n_gpus,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.cfg.test.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predictset,
            batch_size=self.cfg.test.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )
