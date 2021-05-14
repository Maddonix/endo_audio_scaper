from typing import Any, List

import torch
import torch.nn as nn
import torchvision
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import F1, Accuracy, Precision, Recall
from src.models.modules.audio_preprocessing import AudioPreprocess


class AudioLitModel(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        classes: [],
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        data_dir: str = "data/",
        sr: int = 41000,
        duration: int = 10000,  # in ms
        n_mels: int = 64,
        n_fft: int = 1024,
        top_db: int = 80,
        n_mfcc: int = 64,
        hop_len: int = 512,
        **kwargs
    ):
        super().__init__()
        self.data_path = data_dir
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.top_db = top_db
        self.n_mfcc = n_mfcc
        self.hop_len = hop_len
        self.classes = classes
        self.num_classes = len(self.classes)

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # self.model = SimpleConvNet(hparams=self.hparams)
        self.preprocess = AudioPreprocess(hparams=self.hparams)
        self.model = torchvision.models.resnext50_32x4d()
        # Change first layer
        self.model.conv1 = nn.Conv2d(
            1,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False,
        )
        # Change last layer
        self.model.fc = nn.Linear(
            self.model.fc.in_features, self.num_classes, bias=True
        )

        self.model = nn.Sequential(self.preprocess, self.model)

        # loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.train_precision = Precision(
            num_classes=self.num_classes, average=None, multilabel=True
        )
        self.val_precision = Precision(
            num_classes=self.num_classes, average=None, multilabel=True
        )
        self.test_precision = Precision(
            num_classes=self.num_classes, average=None, multilabel=True
        )

        self.train_recall = Recall(
            num_classes=self.num_classes, average=None, multilabel=True
        )
        self.val_recall = Recall(
            num_classes=self.num_classes, average=None, multilabel=True
        )
        self.test_recall = Recall(
            num_classes=self.num_classes, average=None, multilabel=True
        )

        self.train_f1 = F1(num_classes=self.num_classes, average=None, multilabel=True)
        self.val_f1 = F1(num_classes=self.num_classes, average=None, multilabel=True)
        self.test_f1 = F1(num_classes=self.num_classes, average=None, multilabel=True)

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
            "train/f1": [],
            "val/f1": [],
            "train/prec": [],
            "val/prec": [],
            "train/rec": [],
            "val/rec": [],
        }

    def metric_to_dict(self, metric):
        try:
            metric_dict = {self.classes[i]: _ for i, _ in enumerate(metric)}
        except:
            metric_dict = None

        return metric_dict

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.type_as(logits))
        preds = logits.clone().detach()
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        f1 = self.metric_to_dict(self.train_f1(preds, targets))
        prec = self.metric_to_dict(self.train_precision(preds, targets))
        rec = self.metric_to_dict(self.train_recall(preds, targets))
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/prec", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rec", rec, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"]
        )
        self.metric_hist["train/f1"].append(self.trainer.callback_metrics["train/f1"])
        self.metric_hist["train/prec"].append(
            self.trainer.callback_metrics["train/prec"]
        )
        self.metric_hist["train/rec"].append(self.trainer.callback_metrics["train/rec"])
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)
        self.log(
            "train/f1_best",
            self.metric_to_dict(max(self.metric_hist["train/acc"])),
            prog_bar=False,
        )
        self.log(
            "train/prec_best",
            self.metric_to_dict(max(self.metric_hist["train/loss"])),
            prog_bar=False,
        )
        self.log(
            "train/rec_best",
            self.metric_to_dict(max(self.metric_hist["train/loss"])),
            prog_bar=False,
        )

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        f1 = self.metric_to_dict(self.val_f1(preds, targets))
        prec = self.metric_to_dict(self.val_precision(preds, targets))
        rec = self.metric_to_dict(self.val_recall(preds, targets))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/prec", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rec", rec, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.metric_hist["val/f1"].append(self.trainer.callback_metrics["val/f1"])
        self.metric_hist["val/prec"].append(self.trainer.callback_metrics["val/prec"])
        self.metric_hist["val/rec"].append(self.trainer.callback_metrics["val/rec"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)
        self.log(
            "val/f1_best",
            self.metric_to_dict(max(self.metric_hist["val/acc"])),
            prog_bar=False,
        )
        self.log(
            "val/prec_best",
            self.metric_to_dict(max(self.metric_hist["val/loss"])),
            prog_bar=False,
        )
        self.log(
            "val/rec_best",
            self.metric_to_dict(max(self.metric_hist["val/loss"])),
            prog_bar=False,
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        f1 = self.metric_to_dict(self.test_f1(preds, targets))
        prec = self.metric_to_dict(self.test_precision(preds, targets))
        rec = self.metric_to_dict(self.test_recall(preds, targets))
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/prec", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rec", rec, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    # def test_epoch_end(self, outputs: List[Any]):
    #     pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": lr_scheduler,
            # "monitor": "val/loss_best",
        }
