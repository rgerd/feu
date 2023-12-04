from pathlib import Path
from typing import Tuple, cast

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchmetrics
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class PetClassifier(nn.Module):
    """
    Accepts images of dimension 3x256x256 and outputs logits for each of the 37 oxford pet classes.
    Uses MobileNetV2 as a backbone, with a single fully-connected layer on top.
    """

    def __init__(self, classes: list[str]) -> None:
        """
        Initializes the model. Downloads MobileNetV2 weights if necessary.
        """
        super().__init__()
        self.classes = classes
        self.layers = nn.Sequential(
            mobilenet_v2(MobileNet_V2_Weights.DEFAULT),
            nn.Linear(1000, len(self.classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PetClassifierLitModule(L.LightningModule):
    def __init__(
        self,
        classifier: PetClassifier,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self._classes = classifier.classes
        self._num_classes = len(self.classifier.classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> STEP_OUTPUT:
        """
        Trains the model with cross entropy loss.
        """
        input, target = batch
        pred_logits = self.classifier(input)
        target_indices = torch.nn.functional.one_hot(target, num_classes=self._num_classes).float()
        loss = self.loss_fn(pred_logits, target_indices)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        self.pred_labels: list[torch.Tensor] = []
        self.target_labels: list[torch.Tensor] = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> STEP_OUTPUT:
        input, target = batch
        pred_logits = self.classifier(input)

        self.target_labels.append(target)
        self.pred_labels.append(pred_logits.argmax(dim=1))

        accuracy = (self.pred_labels[-1] == self.target_labels[-1]).float().mean()
        self.log_dict({"val_acc": accuracy})
        return None

    def on_validation_epoch_end(self) -> None:
        pred_labels = torch.cat(self.pred_labels)
        target_labels = torch.cat(self.target_labels)

        fig, ax = plt.subplots(figsize=(16, 16))
        cm = torchmetrics.functional.confusion_matrix(
            pred_labels, target_labels, num_classes=self._num_classes, task="multiclass"
        )
        cm = cm / cm.sum(dim=1, keepdim=True)
        cm = cm.cpu().numpy()
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(37))
        ax.set_yticks(range(37))
        ax.set_xticklabels(self._classes, rotation=90)
        ax.set_yticklabels(self._classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        cast(TensorBoardLogger, self.logger).experiment.add_figure(
            "confusion_matrix", fig, global_step=self.current_epoch
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"

    imagenet_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),
            # AutoAugment(AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trainval_dataset = OxfordIIITPet(str(data_dir), download=True, split="trainval", transform=imagenet_transform)
    pet_classes = trainval_dataset.classes

    train_size = int(len(trainval_dataset) * 0.8)
    valid_size = len(trainval_dataset) - train_size
    rand_seed = torch.Generator().manual_seed(43)
    train_set, valid_set = data.random_split(
        dataset=trainval_dataset, lengths=[train_size, valid_size], generator=rand_seed
    )

    train_loader = DataLoader(train_set, shuffle=True, num_workers=7, persistent_workers=True, batch_size=64)
    valid_loader = DataLoader(valid_set, shuffle=False, num_workers=7, persistent_workers=True, batch_size=64)

    classifier = PetClassifierLitModule(PetClassifier(pet_classes))

    trainer = L.Trainer(
        max_epochs=256, check_val_every_n_epoch=2, log_every_n_steps=16, logger=TensorBoardLogger("logs")
    )
    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=valid_loader)
