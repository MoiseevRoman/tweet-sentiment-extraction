import logging
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, get_cosine_schedule_with_warmup
from transformers import logging as transformers_logging

from sentiment_span_extractor.metrics.jaccard import calculate_jaccard_score
from sentiment_span_extractor.models.heads import SpanHead

logger = logging.getLogger(__name__)

# Подавляем предупреждение о неинициализированных весах pooler
# (это нормально для downstream задач, где pooler не используется)
transformers_logging.set_verbosity_error()


class SpanExtractionModule(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str = "roberta-base",
        dropout: float = 0.1,
        lr: float = 3e-5,
        weight_decay: float = 0.001,
        use_last_two_layers: bool = True,
        warmup_steps: int = 0,
        min_words_for_extraction: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.backbone.config.output_hidden_states = True

        hidden_size = self.backbone.config.hidden_size
        if use_last_two_layers:
            hidden_size = hidden_size * 2

        self.head = SpanHead(hidden_size=hidden_size, dropout=dropout)
        self.loss_fn = nn.CrossEntropyLoss()

        self.lr = lr
        self.weight_decay = weight_decay
        self.use_last_two_layers = use_last_two_layers
        self.warmup_steps = warmup_steps
        self.min_words_for_extraction = min_words_for_extraction

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )

        if self.use_last_two_layers:
            hidden_states = torch.cat(
                (outputs.hidden_states[-1], outputs.hidden_states[-2]), dim=-1
            )
        else:
            hidden_states = outputs.last_hidden_state

        start_logits, end_logits = self.head(hidden_states)
        return start_logits, end_logits

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets_start = batch["targets_start"]
        targets_end = batch["targets_end"]

        start_logits, end_logits = self.forward(input_ids, attention_mask)

        start_loss = self.loss_fn(start_logits, targets_start)
        end_loss = self.loss_fn(end_logits, targets_end)
        loss = (start_loss + end_loss) / 2

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets_start = batch["targets_start"]
        targets_end = batch["targets_end"]
        texts = batch["text"]
        sentiments = batch["sentiment"]
        selected_texts = batch["selected_text"]
        offsets = batch["offsets"]

        start_logits, end_logits = self.forward(input_ids, attention_mask)

        start_loss = self.loss_fn(start_logits, targets_start)
        end_loss = self.loss_fn(end_logits, targets_end)
        loss = (start_loss + end_loss) / 2

        start_probs = torch.softmax(start_logits, dim=1)
        end_probs = torch.softmax(end_logits, dim=1)

        jaccard_scores = []
        for i in range(len(texts)):
            idx_start = torch.argmax(start_probs[i]).item()
            idx_end = torch.argmax(end_probs[i]).item()

            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=texts[i],
                target_string=selected_texts[i],
                sentiment_val=sentiments[i],
                idx_start=idx_start,
                idx_end=idx_end,
                offsets=offsets[i].cpu().numpy(),
                min_words_for_extraction=self.min_words_for_extraction,
            )
            jaccard_scores.append(jaccard_score)

        avg_jaccard = torch.tensor(sum(jaccard_scores) / len(jaccard_scores))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_jaccard", avg_jaccard, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_jaccard": avg_jaccard}

    def configure_optimizers(self) -> dict[str, Any]:
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_parameters, lr=self.lr)

        if self.warmup_steps > 0:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return optimizer
