# ACE-Merging

Official implementation of the CVPR 2026 paper:

**"ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation"**

ACE-Merging is a data-free model merging method that adaptively estimates per-task covariance structure to produce a single multi-task model from independently fine-tuned checkpoints -- without access to any training data.

## Overview

Given a pre-trained base model and multiple task-specific fine-tuned models, ACE-Merging:

1. **Computes task vectors** -- the parameter-space difference between each fine-tuned model and the base model.
2. **Estimates per-task covariance** from the task vectors to capture structural heterogeneity across tasks.
3. **Merges via adaptive covariance-weighted averaging**, with an optional low-rank residual correction for highly heterogeneous layers.

The pipeline is fully automated and requires **no data, no gradient computation, and no hyperparameter search**.

## Repository Structure

```
ACE-Merging/
├── main.py                      # Entry point -- runs the full pipeline
├── draw.py                      # Visualization utilities (heterogeneity plots, weight distributions)
├── control/
│   ├── config.py                # Configuration manager (model type, merge method, paths)
│   └── pipeline.py              # Pipeline controller (orchestrates steps 0-3)
├── src/
│   ├── merge/
│   │   ├── strategy.py          # Merging strategies (ACE, Task Arithmetic, TIES, etc.)
│   │   └── sparsify.py          # Sparsification methods (magnitude, rescaled random, etc.)
│   ├── models/
│   │   ├── modeling.py          # ViT image encoder / classifier wrappers (OpenCLIP)
│   │   └── heads.py             # Classification heads
│   ├── datasets/                # Dataset loaders for ViT evaluation (MNIST, CIFAR, Cars, etc.)
│   └── steps/
│       ├── step0.py             # Check for pre-computed task vectors
│       ├── step1.py             # Compute and save task vectors
│       ├── step2.py             # (Reserved) Sparsification
│       └── step3.py             # Merge task vectors and save the merged model
├── evaluation/
│   ├── vit/
│   │   ├── eval_vit.py          # Evaluate merged ViT on image classification benchmarks
│   │   └── get_average_score.py # Aggregate evaluation results
│   └── lm/
│       ├── eval.py              # Common evaluation utilities for LMs
│       ├── eval_bert.py         # Evaluate merged BERT/RoBERTa on GLUE
│       └── eval_gpt.py          # Evaluate merged GPT-2 on GLUE
├── utils/
│   ├── common.py                # Shared helpers (model I/O, task vector utils, seeding)
│   ├── task_vector.py           # TaskVector class (compute, save, load)
│   └── download_checkpoints.py  # Download fine-tuned checkpoints from Google Drive
└── data/                        # (gitignored) Checkpoints, task vectors, configs, merged models
```

## Supported Models

| Model Family | Base Models | Tasks |
|---|---|---|
| **ViT** (OpenCLIP) | ViT-B-32, ViT-B-16, ViT-L-14 | Up to 20 image classification datasets |
| **BERT / RoBERTa** | roberta-base, roberta-large | 8 GLUE tasks |
| **GPT-2** | gpt2 | 7 GLUE tasks |
| **LLaMA** | Llama-3.2-3B-Instruct | Multilingual, Math, Coding |

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+ (with CUDA recommended)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers datasets open_clip_torch easydict tqdm
pip install torchmetrics gdown jstyleson
pip install seaborn matplotlib scipy pandas   # for visualization (draw.py)
```

## Quick Start

### 1. Download Fine-Tuned Checkpoints

For **ViT** models:

```bash
cd utils
python download_checkpoints.py --model 'ViT-B-16' --type 'vit' --kind 'checkpoints'
```

For **RoBERTa** models:

```bash
cd utils
python download_checkpoints.py --model 'roberta-base' --type 'lm' --kind 'checkpoints'
```

For **GPT-2** models, checkpoints are loaded directly from Hugging Face (e.g., `tanganke/gpt2_cola`).

### 2. Configure the Merge

Edit `control/config.py` to set your desired configuration:

```python
class ConfigManager:
    def __init__(self):
        self.with_merge = True
        self.model_type = 'lm'       # 'vit', 'lm', or 'llm'
        self.lm_type = 'gpt2'        # 'gpt2' or 'bert' (when model_type='lm')
        # self.vit_type = 'ViT-B-16' # uncomment for ViT experiments

        self.basic = EasyDict({
            'model_id_list': [
                'gpt2',                    # base model (first entry)
                'tanganke/gpt2_cola',      # fine-tuned models
                'tanganke/gpt2_sst2',
                'tanganke/gpt2_mrpc',
                # ... add more as needed
            ],
        })

        self.merge = EasyDict({
            "method": "ace",              # merging strategy
            "scaling_coefficient": 1.0,
        })
```

### 3. Run the Pipeline

```bash
python main.py
```

The pipeline will:
- **Step 0**: Check for existing task vectors.
- **Step 1**: Compute task vectors for any missing ones.
- **Step 3**: Merge using the configured strategy and save the merged model to `data/models/merged/`.

### 4. Evaluate

**GPT-2 on GLUE:**

```bash
cd evaluation/lm
python eval_gpt.py
```

**BERT/RoBERTa on GLUE:**

```bash
cd evaluation/lm
python eval_bert.py
```

**ViT on image classification:**

```bash
cd evaluation/vit
python eval_vit.py
```

> **Note:** You will need to update the hardcoded model paths in the evaluation scripts to point to your merged model checkpoint.

## Available Merging Strategies

The following strategies are implemented in `src/merge/strategy.py`:

| Method | Config Key | Description |
|---|---|---|
| **ACE-Merging** | `ace` | Adaptive covariance estimation with low-rank residual correction (this paper) |
| Task Arithmetic | `task_arithmetic` | Sum of task vectors |
| Average | `average` | Element-wise mean of task vectors |
| TIES | `ties` | Trim, elect sign, disjoint merge |
| TSVM | `tsvm` | Task-specific vector merging |
| ISO-C | `isoc` | Isotropic singular value replacement |
| ISO-CTS | `isocts` | Common + task-specific subspace decomposition |
| EMR | `emr` | Empirical merging |

## Configuration Reference

All configuration is managed in `control/config.py`:

| Parameter | Description |
|---|---|
| `model_type` | Model family: `'vit'`, `'lm'`, or `'llm'` |
| `lm_type` | Language model type (when `model_type='lm'`): `'gpt2'` or `'bert'` |
| `vit_type` | ViT architecture (when `model_type='vit'`): `'ViT-B-32'`, `'ViT-B-16'`, `'ViT-L-14'` |
| `merge.method` | Merging strategy key (see table above) |
| `merge.scaling_coefficient` | Scaling factor applied to the merged task vector (default: `1.0`) |
| `with_merge` | Enable the merge step (default: `True`) |
| `with_sparisfy` | Enable sparsification before merging (default: `False`) |
| `basic.model_id_list` | List of model identifiers; first entry is the base model |
| `device` | PyTorch device (default: `'cuda:0'`) |

