import os
import sys
import json
import tqdm
import torch
import numpy as np

from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.modeling import ImageClassifier, ImageEncoder, ClassificationHead
from utils.common import apply_to_model

def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)

def eval_single_dataset(model, dataset_name, device='cuda'):
    model.eval()
    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location='../../data/datasets/vit/'
    )

    dataloader = get_dataloader(
        dataset, is_train=False, image_encoder=None, batch_size=128
    )

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    return metrics

all_datasets = ['MNIST', 'Cars', 'DTD', 'EuroSAT', 'GTSRB', 'RESISC45', 'SUN397', 'SVHN', 'PCAM', 'CIFAR100', 'STL10', 'OxfordIIITPet', 'Flowers102', 'FER2013', 'CIFAR10', 'Food101', 'RenderedSST2', 'EMNIST', 'FashionMNIST', 'KMNIST']

all_datasets = all_datasets[:1]

accuracies = {}

model_type = 'ViT-B-16'
# model_type = 'ViT-B-32'
# model_type = 'ViT-L-14'
device = 'cuda'
# pretrained_checkpoint = f'../../data/models/vit/checkpoints/{model_type}/MNISTVal/nonlinear_zeroshot.pt'
# pretrained_model = ImageEncoder.load(model_type, pretrained_checkpoint, device)

merge_ckpt = "/hpc2hdd/home/bx/ACE-Merging/data/models/merged/vit/2026-04-10-21-15-52/model.pt"
image_encoder = torch.load(merge_ckpt, weights_only=False, map_location=device)

# evaluate each task sequentially
for dataset in all_datasets:
    # load pretrained checkpoint

    head_name = f'../../data/models/vit/checkpoints/{model_type}/head_{dataset}Val.pt'
    # model_name = f'../../data/models/vit/checkpoints/{model_type}/{dataset}Val/nonlinear_finetuned.pt'
    classification_head = ClassificationHead.load(head_name, device)
    # image_encoder = ImageEncoder.load(model_type, model_name, device)
    model = ImageClassifier(image_encoder, classification_head)

    for split in ["val", "test"]:
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"
        acc = eval_single_dataset(model, eval_dataset)['top1']
        accuracies[eval_dataset] = acc

model_name = 'wudi_merging_14_tasks'

res_dir_path = f"./results/{model_type}"

os.makedirs(res_dir_path, exist_ok=True)

save_path = os.path.join(res_dir_path, f'{model_name}_results.json')

with open(save_path, "a+") as f:
    f.write(json.dumps(accuracies, sort_keys=False, indent=4) + "\n")

print("File saved at: ", save_path)