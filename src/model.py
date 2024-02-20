from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
# import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from src.submodule import *
from src.utils import dump_final_result
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


class MainModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.bert = Bert_Layer(config)
        # self.classfication_head = RobertaClassificationHead(config)
        self.pooler = Pooler(self.config.pooler_type, config)
        self.dropout = nn.Dropout(p=float(config.dropout))
        # self.cast_1 = nn.Linear(self.config.bert_dim, 256)
        self.cast_1 = PWLayer(self.config.bert_dim, 256)
        # ! parameter whitening may be unstable under this setting
        self.cast_2 = nn.Linear(256, 2)
        # TODO change the final dimension to the #target classes.
        self.gelu = nn.LeakyReLU()

    def forward(self, data):
        news_spans = self.bert(data["input_ids"], data["attention_mask"])
        span_encode = self.pooler(data["attention_mask"], news_spans)

        comb1 = self.gelu(self.cast_1(span_encode))
        comb1 = self.dropout(comb1)
        comb2 = self.gelu(self.cast_2(comb1))
        # comb2 = self.classfication_head(news_spans.last_hidden_state)
        return comb2
    
    def _freeze(self):
        self.bert.freeze_partial()

    def _unfreeze(self):
        self.bert.unfreeze_partial()
        

# main entry point for lightning middle-layer.
class ClicheTeller(L.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = MainModel(config)
        self.validation_step_output = []
        self.validation_step_label = []
        # self.model._freeze()
        self.accmetric = torchmetrics.classification.BinaryAccuracy()
        self.pmetric = torchmetrics.classification.BinaryPrecision()
        self.rmetric = torchmetrics.classification.BinaryRecall()
        self.fmetric = torchmetrics.classification.BinaryF1Score()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target, ids = batch
        # target has shape of [N,]
        outputs = self(inputs)
        # outputs has shape of [N, C]
        loss = F.cross_entropy(outputs, target=target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target, ids = batch
        # target has shape of [N,]
        outputs = self(inputs)
        # outputs has shape of [N, C]
        preds = torch.argmax(outputs, dim=1)
        self.accmetric.update(preds, target)
        self.pmetric.update(preds, target)
        self.rmetric.update(preds, target)
        self.fmetric.update(preds, target)
        return preds
    
    def on_validation_epoch_end(self):
        # https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
        self.log_dict({"evl/ACC": self.accmetric.compute(), "evl/P": self.pmetric.compute(),
                        "evl/R": self.rmetric.compute(), "evl/F1": self.fmetric.compute()}, on_epoch=True, sync_dist=True)
        # print({"ACC": self.accmetric.compute(), "P": self.pmetric.compute(),
        #                 "R": self.rmetric.compute(), "F1": self.fmetric.compute()}, self.global_rank)
        self.accmetric.reset()
        self.pmetric.reset()
        self.fmetric.reset()
        self.rmetric.reset()

    def test_step(self, batch, batch_idx):
        inputs, target, ids = batch
        # target has shape of [N,]
        outputs = self(inputs)
        # outputs has shape of [N, C]
        preds = torch.argmax(outputs, dim=1)
        self.accmetric.update(preds, target)
        self.pmetric.update(preds, target)
        self.rmetric.update(preds, target)
        self.fmetric.update(preds, target)
        return preds
    
    def on_test_epoch_end(self):
        # https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
        self.log_dict({"test/ACC": self.accmetric.compute(), "test/P": self.pmetric.compute(),
                        "test/R": self.rmetric.compute(), "test/F1": self.fmetric.compute()}, on_epoch=True, sync_dist=True)
        # print({"ACC": self.accmetric.compute(), "P": self.pmetric.compute(),
        #                 "R": self.rmetric.compute(), "F1": self.fmetric.compute()}, self.global_rank)
        self.accmetric.reset()
        self.pmetric.reset()
        self.fmetric.reset()
        self.rmetric.reset()

    def predict_step(self, batch, batch_idx):
        inputs, target, ids = batch
        # target has shape of [N,]
        outputs = self(inputs)
        # output has shape of [N, C]
        preds = torch.argmax(outputs, dim=1)
        self.validation_step_output.append(torch.stack((preds, torch.tensor(ids, device=self.device))))
        return (preds, ids)
    
    def on_predict_epoch_end(self):
        all_preds = torch.concat(self.validation_step_output, dim=1)
        self.validation_step_output.clear()
        return dump_final_result(all_preds.T.tolist())

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        # split into two groups, bert related or not related, preparing for separate lr settings.
        bert_related_params = [(n, p) for (n, p) in self.named_parameters() if 'bert' in n]
        nobert_params = [(n, p) for (n, p) in self.named_parameters() if 'bert' not in n]

        optimizer_grouped_parameters = [
            # bert params
            {
                "params": [
                    p for n, p in bert_related_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
                "lr": float(self.config.bert_lr)
            },
            {
                "params": [p for n, p in bert_related_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": float(self.config.bert_lr)
            },
            # no-bert params
            {
                "params": [
                    p for n, p in nobert_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
                "lr": float(self.config.lr) 
            },
            {
                "params": [p for n, p in nobert_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": float(self.config.lr) 
            },
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, eps=float(self.config.adam_epsilon))
        
        total_steps = self.trainer.estimated_stepping_batches

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=int(self.config.warmup_proportion * total_steps),
        #     num_training_steps=total_steps,
        # )
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(self.config.warmup_proportion * total_steps),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
    
