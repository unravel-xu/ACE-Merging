import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from lmDatasets.glue_data_loader import GLUEDataLoader
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    default_data_collator,
    AutoConfig
)

def mrpc_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples['sentence1'],#, 'sentence2'],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def mnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def cola_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question"],
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qqp_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question1"],
        examples["question2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


class TokenizedGLUE:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def load_dataset(
        self, name
    ):
        glue_dataset_loaders = {
            "mrpc": self.load_mrpc_dataset,
            "mnli": self.load_mnli_dataset,
            "cola": self.load_cola_dataset,
            "sst2": self.load_sst2_dataset,
            "qnli": self.load_qnli_dataset,
            "qqp": self.load_qqp_dataset,
            "rte": self.load_rte_dataset
        }
        return glue_dataset_loaders[name]()


    def load_mrpc_dataset(self):
        dataset = load_dataset("glue", "mrpc")
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['sentence1', 'sentence2'],
        )
        return dataset


    def load_rte_dataset(self):
        dataset = load_dataset("glue", "rte", )
        dataset = dataset.map(
            # RTE has the same format as MRPC
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset


    def load_qqp_dataset(self):
        dataset = load_dataset("glue", "qqp")
        dataset = dataset.map(
            partial(qqp_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['question1', 'question2'],
        )
        return dataset


    def load_mnli_dataset(self):
        dataset = load_dataset("glue", "mnli")
        dataset = dataset.map(
            partial(mnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["premise", "hypothesis"],
        )
        return dataset


    def load_cola_dataset(self):
        dataset = load_dataset("glue", "cola")
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset


    def load_sst2_dataset(self):
        dataset = load_dataset("glue", "sst2")
        print(dataset.column_names)
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset


    def load_qnli_dataset(self):
        dataset = load_dataset("glue", "qnli")
        dataset = dataset.map(
            partial(qnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["question", "sentence"],
        )
        return dataset


num_labels = {
        'cola': 2,
        'sst2': 2,
        'mrpc': 2,
        'stsb': 5,
        'qqp': 2,
        'mnli': 3,
        'qnli': 2,
        'rte': 2
    }
dataset_names = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.model_max_length = 512
if tokenizer.pad_token is None:
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)
pretrained_model = GPT2ForSequenceClassification.from_pretrained('gpt2')

models = []
loaders = []
device = 'cuda'
merged_model = GPT2ForSequenceClassification.from_pretrained('/hpc2hdd/home/bx/ACE-Merging/data/models/merged/lm/2026-04-10-20-38-11')
merged_model.to(device)

sum_acc = 0
for dataset_name in dataset_names:
    load_model_path = f"tanganke/gpt2_{dataset_name}"
    finetuned_model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=load_model_path).to(device)
    models.append(finetuned_model)
    merged_model.config = AutoConfig.from_pretrained(load_model_path)
    merged_model.score = finetuned_model.score
    glue = TokenizedGLUE(tokenizer)
    ds = glue.load_dataset(dataset_name)
    try:
        ds_val = ds['validation']
    except:
        ds_val = ds['validation_mismatched']
    with torch.no_grad():
        accuracy = Accuracy("multiclass", num_classes=num_labels[dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
        loader = DataLoader(
            ds_val,
            collate_fn=default_data_collator,
            batch_size=16,
            num_workers=1,
            shuffle=True,
        )
        for batch in (
                pbar := tqdm(
                    loader, desc="Evaluating", leave=False, dynamic_ncols=True
                )
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = merged_model(input_ids, attention_mask=attention_mask)
            # outputs = finetuned_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            acc = accuracy(logits.detach().cpu(), labels.detach().cpu())

        acc = accuracy.compute().item()
        sum_acc += acc
        print(f"{dataset_name}: {acc*100:.2f}%")
print(sum_acc / len(dataset_names))