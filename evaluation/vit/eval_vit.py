"""Evaluate a serialized ImageEncoder saved by main.py on 8, 14, or 20 tasks."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.vit.extended_vit_data import extended_test_dataset
from evaluation.vit.reproduce_vit import load_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.models.modeling import ImageClassifier

ALL_DATASETS = [
    'MNIST', 'Cars', 'DTD', 'EuroSAT', 'GTSRB', 'RESISC45', 'SUN397', 'SVHN',
    'PCAM', 'CIFAR100', 'STL10', 'OxfordIIITPet', 'Flowers102', 'FER2013',
    'CIFAR10', 'Food101', 'RenderedSST2', 'EMNIST', 'FashionMNIST', 'KMNIST',
]


def eval_single_dataset(
    model, dataset_name, device='cuda', *, data_root=ROOT / 'data/datasets/vit',
    batch_size=128, workers=0,
):
    model = model.float().to(device).eval()
    data_root = Path(data_root)
    if dataset_name == 'SUN397' and (data_root / 'sun397/test_manifest.json').exists():
        from torchvision.datasets import ImageFolder
        test_data = ImageFolder(str(data_root / 'sun397/val'), transform=model.val_preprocess)
    elif dataset_name in ALL_DATASETS[8:]:
        test_data = extended_test_dataset(dataset_name, data_root, model.val_preprocess)
    else:
        dataset = get_dataset(
            dataset_name, model.val_preprocess, location=str(data_root),
            batch_size=batch_size, num_workers=workers,
        )
        test_data = dataset.test_loader.dataset

    if len(test_data) == 0:
        raise ValueError(f'{dataset_name} contains no evaluation samples')
    dataset = SimpleNamespace(test_loader=torch.utils.data.DataLoader(test_data))
    dataloader = get_dataloader(
        dataset, is_train=False, batch_size=batch_size,
        num_workers=workers, pin_memory=False,
    )
    correct = total = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=dataset_name):
            batch = maybe_dictionarize(batch)
            images = batch['images'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device)
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise ValueError(f'Non-finite logits on {dataset_name}')
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    if total != len(test_data):
        raise RuntimeError(f'Incomplete evaluation: {total}/{len(test_data)} samples')
    metrics = {'top1': correct / total, 'correct': correct, 'total': total}
    print(f'{dataset_name}: {correct}/{total} = {100 * metrics["top1"]:.4f}%')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged-checkpoint', type=Path, required=True,
                        help='Serialized model.pt from main.py; use the reproduction runner for merged_state.pt')
    parser.add_argument('--model', choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14'], default='ViT-B-32')
    parser.add_argument('--task-count', type=int, choices=[8, 14, 20], default=8)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/datasets/vit')
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'data/models/vit/checkpoints')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['test'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.checkpoints = args.checkpoints.resolve()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    image_encoder = torch.load(args.merged_checkpoint, weights_only=False, map_location=args.device)
    if not isinstance(image_encoder, torch.nn.Module) or not hasattr(image_encoder, 'val_preprocess'):
        raise TypeError('Expected a serialized ImageEncoder from main.py; evaluate merged_state.pt with the reproduction runner')
    image_encoder = image_encoder.float().to(args.device).eval()
    tasks = ALL_DATASETS[:args.task_count]
    accuracies = {}
    for task in tasks:
        head, _ = load_head(args, task)
        model = ImageClassifier(image_encoder, head)
        for split in args.splits:
            name = task if split == 'test' else f'{task}Val'
            accuracies[name] = eval_single_dataset(
                model, name, args.device, data_root=args.data_root,
                batch_size=args.batch_size, workers=args.workers,
            )['top1']
    output = args.output or ROOT / f'evaluation/vit/results/{args.model}/ace_merging_{args.task_count}_tasks_results.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(accuracies, indent=2) + '\n')
    if 'test' in args.splits:
        mean = sum(accuracies[task] for task in tasks) / len(tasks)
        print(f'Complete {len(tasks)}-task test mean: {100 * mean:.4f}%')
    print(f'Saved results to {output.resolve()}')


if __name__ == '__main__':
    main()
