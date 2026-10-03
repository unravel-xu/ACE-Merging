import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from control.config import ConfigManager
from src.merge.strategy import MergeStrategy
from src.steps.step3 import run_step3


class AcePipelineTests(unittest.TestCase):
    def setUp(self):
        first = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10
        second = first.flip(0) * 4 + 0.2
        self.vectors = [
            SimpleNamespace(task_vector_param_dict={"weight": value})
            for value in (first, second)
        ]

    def run_pipeline(self, merge):
        model = torch.nn.Linear(5, 4, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.25)
        config = SimpleNamespace(
            base_model=model,
            model_type="vit",
            device="cpu",
            merge=merge,
            basic=SimpleNamespace(model_id_list=["unused-base"], task_id="ace-test"),
            runtime=SimpleNamespace(task_vectors_path={"first": "first.pt", "second": "second.pt"}),
            save_config=Mock(),
        )
        with (
            patch("src.steps.step3.TaskVector.load", side_effect=self.vectors) as load,
            patch("src.steps.step3.save_merge_model") as save,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_step3(config)
        self.assertEqual(load.call_count, 2)
        self.assertIs(save.call_args.args[-1], model)
        config.save_config.assert_called_once_with()
        return model.weight.detach()

    def test_explicit_parameters_and_scale_reach_the_saved_model(self):
        merge = {
            "method": "ace",
            "scaling_coefficient": 0.83,
            "ace_eps": 7e-4,
            "ace_k_frac": 0.75,
            "ace_tau": 0.04,
        }
        strategy = MergeStrategy(["weight"], self.vectors, "cpu")
        expected_delta = strategy.ace_merging(eps=7e-4, k_frac=0.75, tau=0.04)["weight"]
        legacy_delta = strategy.ace_merging()["weight"]
        self.assertFalse(torch.allclose(expected_delta, legacy_delta))
        actual = self.run_pipeline(merge)
        torch.testing.assert_close(actual, 0.25 + 0.83 * expected_delta)

    def test_legacy_configuration_retains_method_defaults(self):
        expected_delta = MergeStrategy(["weight"], self.vectors, "cpu").ace_merging()["weight"]
        actual = self.run_pipeline({"method": "ace", "scaling_coefficient": 0.9})
        torch.testing.assert_close(actual, 0.25 + 0.9 * expected_delta)

    def test_config_records_ace_parameters_in_saved_json(self):
        with patch("control.config.os.makedirs"):
            config = ConfigManager()
        for key in ("ace_eps", "ace_k_frac", "ace_tau"):
            self.assertIn(key, config.merge)
        config.merge.update(ace_eps=3e-6, ace_k_frac=0.7, ace_tau=0.03)
        with tempfile.TemporaryDirectory() as directory:
            config.basic.config_path = Path(directory) / "config.json"
            config.save_config()
            saved = json.loads(config.basic.config_path.read_text())
        self.assertEqual(saved["merge"]["ace_eps"], 3e-6)
        self.assertEqual(saved["merge"]["ace_k_frac"], 0.7)
        self.assertEqual(saved["merge"]["ace_tau"], 0.03)


if __name__ == "__main__":
    unittest.main()
