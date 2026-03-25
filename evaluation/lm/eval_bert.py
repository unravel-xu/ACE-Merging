import os
import tqdm
import eval
import torch
import jstyleson
import numpy as np
from collections import defaultdict
from transformers import AutoModelForSequenceClassification

def eval_bert(model_id, model_type, dataset_name, device='cuda'):
    """
    Evaluate a single model on a single GLUE dataset.
    """

    # 基准参考值（可选，用于归一化）
    individual = {
        'cola': 64.11, 'mnli': 90.41, 'mrpc': 87.87, 'qnli': 94.21,
        'qqp': 90.35, 'rte': 75.81, 'sst2': 95.99, 'stsb': 90.33
    }

    # 加载对应任务的finetuned模型
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(
        eval.model_path_template.format(name=dataset_name, model_type=model_type)
    ).to(device)

    # 加载待评估模型
    merged_model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)

    # 加载数据集
    data_path = '../../data/datasets/lm/test.json'
    data = jstyleson.load(open(data_path))

    eval_pred = defaultdict(list)

    # 仅选择当前数据集样本
    data_id = eval.glue_data_id_map[dataset_name]

    for data_item in tqdm.tqdm(data, desc=f'infer {dataset_name}'):
        if data_item['dataset_ids'] != data_id:
            continue  # 跳过其他数据集样本

        # 推理
        with torch.no_grad():
            logits = torch.func.functional_call(
                model_finetuned,
                merged_model.state_dict(),
                args=(
                    torch.tensor(data_item['input_ids']).unsqueeze(0).to(device),
                    torch.tensor(data_item['attention_mask']).unsqueeze(0).to(device),
                ),
            ).logits.cpu().numpy()

        eval_pred['predictions'].append(logits)
        eval_pred['label_ids'].append(data_item['label'])

    # 计算指标
    results = eval.compute_single_metrics(
        eval.SimpleNamespace(
            predictions=np.concatenate(eval_pred['predictions']),
            label_ids=np.array(eval_pred['label_ids'])
        ),
        dataset_name
    )

    score = 100 * float(f"{results['averaged_scores']:.4f}")

    print(f"==== {dataset_name.upper()} Evaluation ====")
    print(f"Score: {score:.2f}")
    if dataset_name in individual:
        print(f"Normalized Score: {100 * score / individual[dataset_name]:.2f}")
    print("==============================")

    return score


if __name__ == '__main__':
    model_type = 'roberta-base'
    datasets = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
    total_score = 0
    for dataset_name in datasets:
        # dataset_name = 'mnli'  # 只测这个数据集
        # model_id = f'../../data/models/lm/checkpoints/{model_type}/{dataset_name}/{model_type}_lr1e-05/'
        # model_id = '/hpc2hdd/home/bx/Merge/data/models/merged/lm/2025-10-29-15-37-37'
        model_id = '/hpc2hdd/home/bx/Merge/data/models/merged/lm/2025-10-29-16-47-48'
        total_score += eval_bert(model_id, model_type, dataset_name, device='cuda')
    print(total_score/len(datasets))