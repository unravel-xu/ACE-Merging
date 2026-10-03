import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.vit import eval_vit
from evaluation.vit.get_average_score import DATASETS_8, compute_average
from src.datasets.common import SubsetSampler, get_dataloader
from src.models.modeling import ClassificationHead


class VitEvaluationTests(unittest.TestCase):
    def test_loader_keeps_the_selected_population_and_sample_order(self):
        samples = TensorDataset(torch.arange(5), torch.arange(5))
        original = DataLoader(samples, sampler=SubsetSampler([3, 1]), batch_size=1)
        loader = get_dataloader(SimpleNamespace(test_loader=original), False, batch_size=2)
        self.assertEqual(next(iter(loader))[1].tolist(), [3, 1])

    def test_cli_evaluates_all_eight_tasks_and_writes_valid_json_on_repeat(self):
        # Four real CPU samples, including a final batch of size one.
        samples = TensorDataset(
            torch.tensor([[2., 0.], [0., 2.], [2., 0.], [0., 2.]]),
            torch.tensor([0, 1, 1, 1]),
        )
        dataset = SimpleNamespace(test_loader=DataLoader(samples, batch_size=3))
        encoder = torch.nn.Identity()
        encoder.train_preprocess = encoder.val_preprocess = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / 'model.pt'
            torch.save(encoder, model_path)
            head_root = root / 'checkpoints/ViT-B-32'
            head_root.mkdir(parents=True)
            for task in DATASETS_8:
                if task == 'SVHN':
                    # Some releases omit it; the verified identical MNIST prompts supply this head.
                    continue
                torch.save(ClassificationHead(False, torch.eye(2)), head_root / f'head_{task}Val.pt')
            output = root / 'result.json'
            argv = [
                'eval_vit.py', '--merged-checkpoint', str(model_path),
                '--checkpoints', str(root / 'checkpoints'), '--data-root', str(root / 'datasets'),
                '--device', 'cpu', '--batch-size', '3', '--task-count', '8', '--output', str(output),
            ]
            for _ in range(2):
                with (
                    patch.object(sys, 'argv', argv),
                    patch.object(eval_vit, 'get_dataset', return_value=dataset) as load,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    eval_vit.main()
                self.assertEqual([call.args[0] for call in load.call_args_list], DATASETS_8)
                scores = json.loads(output.read_text())
                self.assertEqual(list(scores), DATASETS_8)
                self.assertEqual(set(scores.values()), {0.75})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compute_average(DATASETS_8, output), (0.75, None))

    def test_mean_rejects_missing_test_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'partial.json'
            output.write_text(json.dumps({task: 1.0 for task in DATASETS_8 if task != 'DTD'}))
            with self.assertRaisesRegex(ValueError, 'missing tasks: DTD'):
                compute_average(DATASETS_8, output)


if __name__ == '__main__':
    unittest.main()
