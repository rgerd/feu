from pathlib import Path
from typing import Tuple

import lightning as L
import torch
import torch.utils.data as data
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.transforms.v2 import AutoAugment, AutoAugmentPolicy


class PetClassifier(nn.Module):
    """
    Accepts images of dimension 3x256x256 and outputs logits for each of the 37 oxford pet classes.
    Uses MobileNetV2 as a backbone, with a single fully-connected layer on top.
    """

    def __init__(self) -> None:
        """
        Initializes the model. Downloads MobileNetV2 weights if necessary.
        """
        super().__init__()
        self.layers = nn.Sequential(
            mobilenet_v2(MobileNet_V2_Weights.DEFAULT),
            nn.Dropout(p=0.2),
            nn.Linear(1000, 37),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PetClassifierLitModule(L.LightningModule):
    def __init__(self, classifier: PetClassifier) -> None:
        super().__init__()
        self.classifier = classifier
        self.training_augment = transforms.Compose(
            [
                AutoAugment(AutoAugmentPolicy.IMAGENET),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> STEP_OUTPUT:
        """
        Trains the model with cross entropy loss.
        """
        input, target = batch
        input = self.training_augment(input)
        pred_logits = self.classifier(input)
        target_indices = torch.nn.functional.one_hot(target, num_classes=37).float()
        loss = self.loss_fn(pred_logits, target_indices)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> STEP_OUTPUT:
        input, target = batch
        pred_logits = self.classifier(input)
        accuracy = (pred_logits.argmax(dim=1) == target).float().mean()
        self.log_dict({"val_acc": accuracy})
        return None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    dataset = OxfordIIITPet(
        str(data_dir),
        download=True,
        split="trainval",
        transform=transforms.Compose(
            [
                transforms.Resize((256, 256), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size
    rand_seed = torch.Generator().manual_seed(43)
    train_set, valid_set = data.random_split(dataset=dataset, lengths=[train_size, valid_size], generator=rand_seed)

    train_loader = DataLoader(train_set, shuffle=True, num_workers=7, persistent_workers=True, batch_size=64)
    valid_loader = DataLoader(valid_set, shuffle=False, num_workers=7, persistent_workers=True, batch_size=64)

    classifier = PetClassifierLitModule(PetClassifier())

    trainer = L.Trainer(max_epochs=256, check_val_every_n_epoch=2, log_every_n_steps=16)
    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=valid_loader)
