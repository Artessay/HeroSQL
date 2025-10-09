import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F

from src.model import Plain, Satie, Syntax, Lope, Hero

class TripleFusionLightningModule(pl.LightningModule):
    def __init__(
            self, 
            model_name: str = "hero",
            embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
            hidden_dim: int = 128,
            projection_dim: int = 64,
            dropout: float = 0.3,
            operator_embedding_dim: int = 16,
            gnn_num_layers: int = 3,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(
            "model_name",
            "learning_rate", "weight_decay", 
            "dropout",
            "embedding_model_name",
            "operator_embedding_dim",
            "projection_dim", "hidden_dim",
            "gnn_num_layers",
        )

        if model_name == "hero":
            self.model = Hero(
                embedding_model_name=embedding_model_name,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
                dropout=dropout,
                gnn_num_layers=gnn_num_layers,
            )
        elif model_name == "lope":
            self.model = Lope(
                embedding_model_name=embedding_model_name,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
                dropout=dropout,
                gnn_num_layers=gnn_num_layers,
            )
        elif model_name == "syntax":
            self.model = Syntax(
                embedding_model_name=embedding_model_name,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
                dropout=dropout,
                gnn_num_layers=gnn_num_layers,
            )
        elif model_name == "satie":
            self.model = Satie(
                embedding_model_name=embedding_model_name,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
                dropout=dropout,
                operator_embedding_dim=operator_embedding_dim,
                gnn_num_layers=gnn_num_layers,
            )
        elif model_name == "plain":
            self.model = Plain(
                embedding_model_name=embedding_model_name,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
                dropout=dropout,
            )

    def setup(self, stage):
        # Prepare metrics
        self.metrics_acc = torchmetrics.Accuracy(task="binary")
        self.metrics_precision = torchmetrics.Precision(task="binary")
        self.metrics_recall = torchmetrics.Recall(task="binary")
        self.metrics_f1 = torchmetrics.F1Score(task="binary")
        self.metrics_auc_roc = torchmetrics.AUROC(task="binary")
        self.metrics_auc_pr = torchmetrics.AveragePrecision(task="binary")
        
        # Prepare test storage (for predictions and labels)
        self.test_probs_batches = []
        self.test_labels_batches = []

        return super().setup(stage)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels").float()
        logits = self(**batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        batch_size = len(labels)
        self.log("train_loss", loss, batch_size=batch_size)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels").float()
        logits = self(**batch)
        val_loss = F.binary_cross_entropy_with_logits(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        labels = labels.int()
        batch_size = len(labels)
        self.log("val_loss", val_loss, batch_size=batch_size, prog_bar=True, sync_dist=True)

        self.metrics_acc.update(1 - preds, 1 - labels)
        self.metrics_precision.update(1 - preds, 1 - labels)
        self.metrics_recall.update(1 - preds, 1 - labels)
        self.metrics_f1.update(1 - preds, 1 - labels)
        self.metrics_auc_roc.update(1 - probs, 1 - labels)
        self.metrics_auc_pr.update(1 - probs, 1 - labels)

        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        # log metrics
        self.log("val_accuracy",  self.metrics_acc.compute(),       sync_dist=True)
        self.log("val_precision", self.metrics_precision.compute(), sync_dist=True)
        self.log("val_recall",    self.metrics_recall.compute(),    sync_dist=True)
        self.log("val_f1",        self.metrics_f1.compute(),        sync_dist=True)
        self.log("val_auc_roc",   self.metrics_auc_roc.compute(),   sync_dist=True)
        self.log("val_auc_pr",    self.metrics_auc_pr.compute(),    sync_dist=True)

        # reset metrics
        self.metrics_acc.reset()
        self.metrics_precision.reset()
        self.metrics_recall.reset()
        self.metrics_f1.reset()
        self.metrics_auc_roc.reset()
        self.metrics_auc_pr.reset()

    def test_step(self, batch, batch_idx):
        # Separate labels from batch and generate logits
        labels = batch.pop("labels").int()
        logits = self(**batch)
        # Compute sigmoid probabilities for binary classification
        probs = torch.sigmoid(logits)
        # Threshold probabilities to get predictions (0/1)
        preds = (probs > 0.5).int()
        
        # Update various test metrics
        self.metrics_acc.update(1 - preds, 1 - labels)
        self.metrics_precision.update(1 - preds, 1 - labels)
        self.metrics_recall.update(1 - preds, 1 - labels)
        self.metrics_f1.update(1 - preds, 1 - labels)
        self.metrics_auc_roc.update(1 - probs, 1 - labels)
        self.metrics_auc_pr.update(1 - probs, 1 - labels)
        
        # Save probabilities and labels to instance batches list
        self.test_probs_batches.append(probs.detach().cpu())
        self.test_labels_batches.append(labels.detach().cpu())
        
    def on_test_epoch_end(self):
        # log metrics
        self.log("test_accuracy",  self.metrics_acc.compute(),       sync_dist=True)
        self.log("test_precision", self.metrics_precision.compute(), sync_dist=True)
        self.log("test_recall",    self.metrics_recall.compute(),    sync_dist=True)
        self.log("test_f1",        self.metrics_f1.compute(),        sync_dist=True)
        self.log("test_auc_roc",   self.metrics_auc_roc.compute(),   sync_dist=True)
        self.log("test_auc_pr",    self.metrics_auc_pr.compute(),    sync_dist=True)

        # reset metrics
        self.metrics_acc.reset()
        self.metrics_precision.reset()
        self.metrics_recall.reset()
        self.metrics_f1.reset()
        self.metrics_auc_roc.reset()
        self.metrics_auc_pr.reset()

        # Concatenate all batches to get complete test set inference results
        probs_cat = torch.cat(self.test_probs_batches, dim=0)
        labels_cat = torch.cat(self.test_labels_batches, dim=0)
        # Convert tensors to python lists
        self.test_probs = probs_cat.numpy().tolist()
        self.test_labels = labels_cat.numpy().tolist()

        self.test_probs_batches.clear()
        self.test_labels_batches.clear()

        return self.test_probs, self.test_labels

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
    