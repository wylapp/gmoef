import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
import os
from os import path
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, Features, Value, ClassLabel
from transformers import AutoTokenizer, DataCollatorWithPadding
import pickle

class M4DataModule(L.LightningDataModule):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)
        self.collator = DataCollatorWithPadding(self.tokenizer)
        # * Create the meta-info manually
        self.features = Features({"text": Value(dtype="string"), "label": ClassLabel(num_classes=2),
                                  "id": Value(dtype='uint32'), "model": Value(dtype='string'), "source": Value(dtype='string')})
        
        self.features2 = Features({"text": Value(dtype="string"), 
                                  "id": Value(dtype='uint32')})
        assert self.config.task in ('a1', 'a2'), "This DataModule is only designed for task a1 and a2!"
        self.lingual = 'multilingual' if self.config.task=='a1' else 'monolingual'

    
    def collate(self, batch):
        input_ids = []
        masks = []
        label = []
        ids = []
        for x in batch:
            input_ids.append(x["input_ids"])
            masks.append(x["attention_mask"])
            label.append(x["label"])
            ids.append(x["id"])
        sec1 = self.collator({"input_ids": input_ids, "attention_mask": masks})

        return sec1, torch.tensor(label), ids
    
    def collate_test(self, batch):
        input_ids = []
        masks = []
        label = []
        ids = []
        for x in batch:
            input_ids.append(x["input_ids"])
            masks.append(x["attention_mask"])
            ids.append(x["id"])
        sec1 = self.collator({"input_ids": input_ids, "attention_mask": masks})

        return sec1, torch.tensor(label), ids

    def preprocess_function(self, examples, **fn_kwargs):
        # ! the MAXIMUM input length for BERT(RoBERTa) is 512! Do not exceed the limit!
        return fn_kwargs['tokenizer'](examples["text"], truncation=True)

    def prepare_data(self) -> None:
        # ! preprocess, only call once in the main process
        lis_files = os.listdir(self.config.data_dir)
        if f"subtaskA_train_{self.lingual}.pkl" in lis_files and f"subtaskA_dev_{self.lingual}.pkl" in lis_files and f"subtaskA_test_{self.lingual}.pkl" in lis_files and self.config.force_new == False:
            pass
        else:
            train_df = pd.read_json(path.join(self.config.data_dir, f"subtaskA_train_{self.lingual}.jsonl"), lines=True)
            test_df = pd.read_json(path.join(self.config.data_dir, f"subtaskA_dev_{self.lingual}.jsonl"), lines=True)
            pred_df = pd.read_json(path.join(self.config.data_dir, f"subtaskA_test_{self.lingual}.jsonl"), lines=True)
            train_ds = Dataset.from_pandas(train_df, features=self.features)
            test_ds = Dataset.from_pandas(test_df, features=self.features)
            pred_ds = Dataset.from_pandas(pred_df, features=self.features2)
            tokenized_train_ds = train_ds.map(self.preprocess_function, batched=True, fn_kwargs={'tokenizer': self.tokenizer})
            tokenized_test_ds = test_ds.map(self.preprocess_function, batched=True, fn_kwargs={'tokenizer': self.tokenizer})
            tokenized_pred_ds = pred_ds.map(self.preprocess_function, batched=True, fn_kwargs={'tokenizer': self.tokenizer})
            with open(path.join(self.config.data_dir, f"subtaskA_train_{self.lingual}.pkl"), "wb") as pf:
                pickle.dump(tokenized_train_ds, pf)

            with open(path.join(self.config.data_dir, f"subtaskA_dev_{self.lingual}.pkl"), "wb") as pf:
                pickle.dump(tokenized_test_ds, pf)

            with open(path.join(self.config.data_dir, f"subtaskA_test_{self.lingual}.pkl"), "wb") as pf:
                pickle.dump(tokenized_pred_ds, pf)

        

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage == 'validate':
            with open(path.join(self.config.data_dir, f"subtaskA_train_{self.lingual}.pkl"), "rb") as pf:
                tokenized_train_ds = pickle.load(pf)
            with open(path.join(self.config.data_dir, f"subtaskA_dev_{self.lingual}.pkl"), "rb") as pf:
                tokenized_test_ds = pickle.load(pf)
            # ! DO NOT use the `sklearn.model_selection.train_test_split` here!
            # splited = tokenized_train_ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=self.config.seed)
            self.train_ds = tokenized_train_ds
            self.val_ds = tokenized_test_ds
        if stage == 'test':
            with open(path.join(self.config.data_dir, f"subtaskA_dev_{self.lingual}.pkl"), "rb") as pf:
                tokenized_test_ds = pickle.load(pf)
            self.test_ds = tokenized_test_ds
        if stage == 'predict':
            with open(path.join(self.config.data_dir, f"subtaskA_test_{self.lingual}.pkl"), "rb") as pf:
                tokenized_pred_ds = pickle.load(pf)
            self.pred_ds = tokenized_pred_ds
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.batch_size, collate_fn=self.collate)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.config.batch_size, collate_fn=self.collate)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_ds, batch_size=self.config.batch_size, collate_fn=self.collate_test)
    
    @property
    def train_len(self):
        return len(self.train_ds)
        

    

    