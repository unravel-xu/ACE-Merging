# ViT reproduction settings

These settings were checked in additional reproduction runs using the historical
ACE/TSV checkpoints. They are newly verified configurations, not reconstructed
original experiment command lines. See [dataset preparation](vit_datasets.md)
before running evaluation.

## Verified configurations

Accuracy is the unweighted mean of per-task accuracies on complete test sets.
Runs used FP32, QuickGELU, seed 0, and disabled AMP/TF32. The reference is
[ACE-Merging Table 7](https://arxiv.org/html/2603.02945v2).

| Model | Tasks | `scaling_coefficient` | `ace_eps` | `ace_k_frac` | `ace_tau` | Reproduced mean (%) | Paper mean (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViT-B-32 | 8 | 1.0 | 1e-5 | 0.5 | 0.03 | 88.5174 | 87.88 |
| ViT-B-32 | 14 | 1.0 | 1e-5 | 0.5 | 0.03 | 83.6653 | 82.29 |
| ViT-B-32 | 20 | 1.0 | 1e-5 | 0.5 | 0.03 | 80.3598 | 78.83 |
| ViT-B-16 | 8 | 1.0 | 1e-5 | 0.5 | 0.03 | 91.3526 | 90.60 |
| ViT-B-16 | 14 | 1.0 | 1e-5 | 0.5 | 0.03 | 87.0789 | 86.12 |
| ViT-B-16 | 20 | 1.0 | 1e-5 | 0.5 | 0.03 | 83.9143 | 82.11 |
| ViT-L-14 | 8 | 0.95 | 1e-5 | 0.7 | 0.03 | 94.7256 | 94.57 |
| ViT-L-14 | 14 | 0.95 | 1e-5 | 0.7 | 0.03 | 91.6343 | 91.07 |
| ViT-L-14 | 20 | 0.95 | 3e-6 | 0.7 | 0.03 | 90.8238 | 89.49 |

The ViT-B-32 eight-task row is the independently confirmed configuration; the
ViT-L-14 eight-task row uses the validation-selected `ace_tau=0.03` configuration.
These are distinct from the highest observed test scores at scale 0.9 and tau
0.04, respectively. Reaching the reported mean does not imply reaching every
individual task's reported accuracy.

## Configure the main pipeline

Set the fields in `control/config.py`, using the row for the model and task count:

```python
self.merge = EasyDict({
    "method": "ace",
    "scaling_coefficient": 1.0,
    "ace_eps": 1e-5,
    "ace_k_frac": 0.5,
    "ace_tau": 0.03,
})
```

The ACE branch in `src/steps/step3.py` passes these fields to the method:

```python
merged_task_vector = ms.ace_merging(
    eps=config.merge.get("ace_eps", 1e-2),
    k_frac=config.merge.get("ace_k_frac", 0.3),
    tau=config.merge.get("ace_tau", 0.3),
)
```

`scaling_coefficient` is already passed to `apply_to_model`. New ViT configurations
use the ViT-B profile shown above; set the ViT-L fields explicitly when changing
the backbone. Configurations that omit the three ACE fields retain the previous
method defaults (`eps=1e-2`, `k_frac=0.3`, `tau=0.3`). Non-ViT configurations keep
those previous defaults as well.

The model list must contain the base checkpoint followed by the matching 8, 14,
or 20 experts. Their order is:

```text
MNIST, Cars, DTD, EuroSAT, GTSRB, RESISC45, SUN397, SVHN,
PCAM, CIFAR100, STL10, OxfordIIITPet, Flowers102, FER2013,
CIFAR10, Food101, RenderedSST2, EMNIST, FashionMNIST, KMNIST
```

## Explicit FP32 reproduction commands

Run from the repository root in the `ace-merging` environment. The explicit
runner controls QuickGELU and FP32 rather than relying on OpenCLIP's constructor
defaults. The activation cannot be inferred from checkpoint tensors alone.
The checked environment used Python 3.10, PyTorch 2.4.0, torchvision 0.19.0,
and `open_clip_torch` 3.3.0. In addition to the main dependencies, the preparation
and evaluation helpers use `scipy`, `pyarrow`, `h5py`, `requests`, and
`beautifulsoup4`.

First prepare the datasets as described in [the dataset guide](vit_datasets.md).
Place the complete historical model/head release in a dedicated namespace. For
example, download the [historical ViT-L-14 release](https://drive.google.com/drive/folders/10DyLk0jvPB0O-ffU2x1_FGQ2DNJ-_-PY)
and arrange it as:

```text
data/models/vit/checkpoints_legacy/ViT-L-14/
  MNISTVal/nonlinear_zeroshot.pt
  MNISTVal/nonlinear_finetuned.pt
  CarsVal/nonlinear_finetuned.pt
  DTDVal/nonlinear_finetuned.pt
  ...
  head_MNISTVal.pt
  head_CarsVal.pt
  head_DTDVal.pt
  ...
```

An eight-task ViT-L-14 run using the validation-selected configuration is:

```bash
conda activate ace-merging
python evaluation/vit/reproduce_vit.py \
  --phase merge --model ViT-L-14 --task-count 8 \
  --checkpoints data/models/vit/checkpoints_legacy \
  --data-root data/datasets/vit_reproduction \
  --activation quickgelu \
  --scaling-coefficient 0.95 --eps 1e-5 --k-frac 0.7 --tau 0.03 \
  --output data/reproductions/documented_l14_8task

python evaluation/vit/evaluate_vit_runs_bounded.py \
  --model ViT-L-14 \
  --checkpoints data/models/vit/checkpoints_legacy \
  --data-root data/datasets/vit_reproduction \
  --outputs data/reproductions/documented_l14_8task \
  --batch-size 128 --preprocess-workers 2
```

For other rows, change the model, task count, four parameters, and output path.
Use a new output directory for each merge. Complete 14/20-task runs require the
additional datasets in the guide. The evaluator uses a temporary disk cache to
limit RAM use and records all predictions, sample counts, input hashes, and the
complete-test mean in `results.json`. Evaluation resumes completed tasks;
repeating a merge in an occupied output directory is rejected.

The generic `main.py` pipeline now forwards the ACE parameters. For the reported
precision and activation settings, use the explicit runner above; the legacy
model constructor does not explicitly force QuickGELU.

## DTD checkpoint provenance

[TSV's README](https://github.com/AntoAndGar/task_singular_vectors#checkpoints)
identifies its release as an earlier version of TALL Masks checkpoints. TALL
Masks later [acknowledged a DTD split-generation problem](https://github.com/nik-dim/tall_masks/issues/1#issuecomment-2361419293)
and [announced updated checkpoints](https://github.com/nik-dim/tall_masks/issues/1#issuecomment-2541695596).

The historical ViT-L-14 release and the corrected upstream release have 41
model/head files. Only the DTD model file differs in our complete comparison.

| DTD checkpoint | Full-file SHA256 |
|---|---|
| [Historical ACE/TSV ViT-L-14](https://drive.google.com/file/d/17TnemtBH3ec1IwHBo7RxJ_mMegYlgZLE/view) | `3106e4e4100e1bf23289d0da18c86bb97e3ecf069a6da0d8abecb9b561691413` |
| Corrected `nik-dim/tall_masks`, revision `c8d352f3802fc221a0e993fe6335ed03dfe91d73` | `597ba6d883cf1cf3aafc7c8ed62378e0e38eaea391aa6404c845a5a3a576a33e` |

On the same official DTD `test1.txt` images and with the same classification head,
the historical expert scores 98.510638% and the corrected expert scores 84.414894%.
Both versions must be labeled explicitly. To evaluate the corrected version,
prepare it under a different checkpoint root and use a separate output directory.

The highest completed corrected-DTD ViT-L-14 results so far are:

| Tasks | Scale | Eps | K fraction | Tau | Corrected mean (%) | Paper mean (%) | Difference (percentage points) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.95 | 1e-5 | 0.7 | 0.04 | 92.9318 | 94.57 | -1.6382 |
| 14 | 0.95 | 1e-5 | 0.7 | 0.03 | 90.6397 | 91.07 | -0.4303 |
| 20 | 1.0 | 1e-5 | 0.5 | 0.03 | 89.7283 | 89.49 | +0.2383 |

These are separately tested configurations, not a comparison holding all
hyperparameters fixed. Corrected-DTD ViT-B-32/ViT-B-16 reproduction results have
not yet been established. All main DTD evaluations use official `test1.txt`;
corrected weights do not mean changing the evaluation to official split 2.

## Scope of the reproduction claim

The parameter search was exploratory, with earlier test results already
available. ViT-L-14 validation selection used seven complete validation sets,
excluding SUN397; the eight-task configuration was transferred to fourteen
tasks. For twenty tasks, thirteen tasks were not part of validation selection.
EuroSAT/SUN397 image assignments have not been confirmed to match the paper
exactly. The historical DTD expert's original training-image manifest has not
been reconstructed, so no particular pattern of training/test overlap is
asserted. Numerical agreement with the historical release is therefore reported
separately from evaluation with the corrected checkpoint.
