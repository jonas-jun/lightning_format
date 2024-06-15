import torch.nn as nn
from transformers import CLIPModel


class accommClassifier(nn.Module):
    def __init__(self, cfg):
        super(accommClassifier, self).__init__()
        self.cfg = cfg
        self.clip = CLIPModel.from_pretrained(cfg.model.name, cache_dir=cfg.model.ckpt)
        self.clip = self.clip.vision_model
        hid_dim = self.clip.encoder.layers[-1].mlp.fc2.out_features
        # self.clip.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(hid_dim, 512),
            nn.ReLU(),
            nn.Linear(512, cfg.model.n_classes),
        )

    def forward(self, images):
        out = self.clip(images).pooler_output
        out = self.head(out)
        return out
