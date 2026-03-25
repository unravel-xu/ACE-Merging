import os
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from lmDatasets.glue_data_loader import GLUEDataLoader, glue_data_num_labels_map, rev_glue_data_id_map
from lmDatasets.glue_metrics import compute_metrics

class CustomizedTrainer(Trainer):

    def __init__(self, use_multitask_setting: bool = False, *args, **kwargs):
        """
        Customized trainer with user-defined train loss function.
        :param use_multitask_setting: boolean, whether to use multitask setting
        """
        super(CustomizedTrainer, self).__init__(*args, **kwargs)
        self.use_multitask_setting = use_multitask_setting

    def compute_loss(self, model: nn.Module, inputs: dict, return_outputs: bool = False):
        """
        how the loss is computed by CustomizedTrainer
        :param model: nn.Module
        :param inputs: dict, model inputs
        :param return_outputs: boolean, whether return the outputs or not
        :return:
        """
        assert "labels" in inputs, "labels are not involved in inputs!"
        labels = inputs.pop("labels")
        if self.use_multitask_setting:
            assert "dataset_ids" in inputs.keys(), "key dataset_ids is missing in the inputs!"
            # Tensor
            dataset_ids = inputs["dataset_ids"]
            outputs = model(**inputs)
            logits = outputs["logits"]
            total_loss = None
            for dataset_id in dataset_ids.unique():
                single_dataset_indices = dataset_ids == dataset_id
                single_dataset_num_labels = glue_data_num_labels_map[rev_glue_data_id_map[dataset_id.item()]]
                # cross-entropy loss for classification
                if single_dataset_num_labels > 1:
                    loss = F.cross_entropy(input=logits[single_dataset_indices][:, :single_dataset_num_labels], target=labels[single_dataset_indices].long())
                # mse loss for regression
                else:
                    assert single_dataset_num_labels == 1, "wrong number of labels!"
                    loss = F.mse_loss(input=logits[single_dataset_indices][:, 0], target=labels[single_dataset_indices])
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
            return (total_loss, outputs) if return_outputs else total_loss
        else:
            outputs = model(**inputs)
            logits = outputs["logits"]
            if logits.shape[1] > 1:
                # cross-entropy loss for classification
                loss = F.cross_entropy(input=logits, target=labels)
            else:
                # mse loss for regression
                assert logits.shape[1] == 1, "wrong number of labels!"
                loss = F.mse_loss(input=logits.squeeze(dim=1), target=labels)
            return (loss, outputs) if return_outputs else loss


dataset_acc_dict = {
    "cola": "eval_matthews_correlation",
    "sst2": "eval_accuracy",
    "mrpc": "eval_accuracy",
    "stsb": "eval_averaged_scores",
    "qqp": "eval_accuracy",
    "mnli": "eval_accuracy",
    "qnli": "eval_accuracy",
    "rte": "eval_accuracy"
}

def eval_lm_on_dataset(model_id, dataset_name, device='cuda'):
    batch_size = 16
    base_model_id = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)
    models_to_merge, trainers, = [], []
    merged_model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    ft_model_id = f'EvaristeL/roberta-base-{dataset_name}-sft'
    train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                    train_split_ratio_for_val=0.1,
                                                                                    max_seq_length=128)
    training_args = TrainingArguments(
        output_dir='./results/',            # save model directory
        per_device_train_batch_size=batch_size,       # batch size per device during training
        per_device_eval_batch_size=batch_size,        # batch size for evaluation
        report_to="none"
    )

    ft_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=ft_model_id,
        num_labels=num_labels).to(device)

    trainer = CustomizedTrainer(
        model=merged_model,               # model to be merged
        args=training_args,                 # training arguments
        train_dataset=train_dataset,        # training dataset
        eval_dataset=test_dataset,          # evaluation dataset
        compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
        tokenizer=tokenizer                 # tokenizer
    )

    models_to_merge.append(ft_model.to(device))
    trainers.append(trainer)
    merged_model.classifier = ft_model.classifier
    test_metrics = trainer.evaluate()
    test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
    return test_metrics
    

dataset_names = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]

rs = []
avg_score = 0.0
for dataset_name in dataset_names:
    # model_id = f"EvaristeL/roberta-base-{dataset_name}-sft"
    model_id = '/hpc2hdd/home/bx/Merge/data/models/merged/lm/2025-10-28-18-47-31'
    metrics = eval_lm_on_dataset(model_id, dataset_name)
    score = metrics[dataset_acc_dict[dataset_name]]
    print(score)
    avg_score += score
    rs.append({"model_id": model_id, "dataset": dataset_name, "score": score})

print("\n\n")
for r in rs:
    print(f"{r['dataset']} - {r['score']}")
print(f'{avg_score / len(dataset_names):.4f}')