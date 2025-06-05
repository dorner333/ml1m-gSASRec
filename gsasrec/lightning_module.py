from pathlib import Path
import pytorch_lightning as pl
import torch

from gsasrec.dataset_utils import (
    get_num_items,
    get_train_dataloader,
    get_val_dataloader,
)
from gsasrec.eval_utils import evaluate_pt
from gsasrec.utils import build_model


class GSASRecLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = build_model(config)
        self.num_items = get_num_items(config.dataset_name)
        self.val_dataloader_obj = get_val_dataloader(
            config.dataset_name,
            batch_size=config.eval_batch_size,
            max_length=config.sequence_length,
        )
        self.best_metric = float("-inf")
        self.steps_not_improved = 0

        self.train_losses = []
        self.val_ndcg10 = []
        self.val_r1 = []
        self.val_r10 = []

    def forward(self, input_seq):
        return self.model(input_seq)

    def training_step(self, batch, batch_idx):
        positives, negatives = batch
        positives = positives.to(self.device)
        negatives = negatives.to(self.device)

        model_input = positives[:, :-1]
        last_hidden_state, _ = self.model(model_input)
        labels = positives[:, 1:]
        negatives = negatives[:, 1:, :]

        pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)
        output_embeddings = self.model.get_output_embeddings()
        pos_neg_embeddings = output_embeddings(pos_neg_concat)

        mask = (model_input != self.num_items + 1).float()
        logits = torch.einsum("bse,bsne->bsn", last_hidden_state, pos_neg_embeddings)

        gt = torch.zeros_like(logits)
        gt[:, :, 0] = 1

        alpha = self.config.negs_per_pos / (self.num_items - 1)
        t = self.config.gbce_t
        beta = alpha * ((1 - 1 / alpha) * t + 1 / alpha)

        positive_logits = logits[:, :, 0:1].to(torch.float64)
        negative_logits = logits[:, :, 1:].to(torch.float64)

        eps = 1e-10
        positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1 - eps)
        positive_probs_adjusted = torch.clamp(
            positive_probs.pow(-beta), 1 + eps, torch.finfo(torch.float64).max
        )
        to_log = torch.clamp(
            torch.div(1.0, (positive_probs_adjusted - 1)),
            eps,
            torch.finfo(torch.float64).max,
        )
        positive_logits_transformed = to_log.log()
        logits = torch.cat([positive_logits_transformed, negative_logits], dim=-1)

        loss_per_element = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, gt, reduction="none"
            ).mean(-1)
            * mask
        )
        loss = loss_per_element.sum() / mask.sum()

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        losses = [out["loss"] for out in outputs]
        avg_train_loss = torch.stack(losses).mean().item()
        self.train_losses.append(avg_train_loss)
        self.log("train_loss", avg_train_loss, on_epoch=True, prog_bar=True)

        result = evaluate_pt(
            self.model,
            self.val_dataloader_obj,
            self.config.metrics,
            self.config.recommendation_limit,
            self.config.filter_rated,
            device=self.device,
        )
        ndcg10 = result[self.config.val_metric]
        r1 = result["R@1"]
        r10 = result["R@10"]

        self.val_ndcg10.append(ndcg10)
        self.val_r1.append(r1)
        self.val_r10.append(r10)

        self.log("val_nDCG10", ndcg10, on_epoch=True, prog_bar=True)
        self.log("val_R1", r1, on_epoch=True, prog_bar=False)
        self.log("val_R10", r10, on_epoch=True, prog_bar=False)

        if ndcg10 > self.best_metric:
            self.best_metric = ndcg10
            self.steps_not_improved = 0
            model_path = Path(
                f"models/gsasrec-"
                f"{self.config.dataset_name}-"
                f"step:{self.global_step}-"
                f"t:{self.config.gbce_t}-"
                f"negs:{self.config.negs_per_pos}-"
                f"emb:{self.config.embedding_dim}-"
                f"dropout:{self.config.dropout_rate}-"
                f"metric:{self.best_metric}.pt"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(self, "best_model_name"):
                try:
                    Path(self.best_model_name).unlink()
                except FileNotFoundError:
                    pass
            self.best_model_name = str(model_path)
            torch.save(self.model.state_dict(), str(model_path))
        else:
            self.steps_not_improved += 1
            if self.steps_not_improved >= self.config.early_stopping_patience:
                self.trainer.should_stop = True

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def train_dataloader(self):
        return get_train_dataloader(
            self.config.dataset_name,
            batch_size=self.config.train_batch_size,
            max_length=self.config.sequence_length,
            train_neg_per_positive=self.config.negs_per_pos,
        )
