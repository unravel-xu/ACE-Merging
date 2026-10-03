# Preparing ViT datasets

Dataset images and partition files are not bundled with the repository. Loaders
are in `src/datasets`; some download their files, while others read an already
prepared directory. Use one explicit dataset root for preparation and evaluation.
The upstream protocol is described in [Task Arithmetic issue #1](https://github.com/mlfoundations/task_vectors/issues/1).

## Required directory layout

```text
DATA_ROOT/
  stanford_cars/
    devkit/cars_train_annos.mat
    devkit/cars_meta.mat
    cars_test_annos_withlabels.mat
    cars_train/*.jpg
    cars_test/*.jpg
  dtd/
    train/<class>/*.jpg
    val/<class>/*.jpg
  sun397/
    train/<class>/*.jpg
    val/<class>/*.jpg
  EuroSAT_splits/
    train/<class>/*.jpg
    val/<class>/*.jpg
    test/<class>/*.jpg
  resisc45/
    NWPU-RESISC45/<class>/*.jpg
    resisc45-train.txt
    resisc45-val.txt
    resisc45-test.txt
```

The local Stanford Cars reader also accepts
`stanford_cars/devkit/cars_test_annos_withlabels.mat`. Keep the official labels
and class order when using a prepared copy of a dataset. Downloading images alone
does not supply the annotation or split-list files needed by these loaders.

## DTD: official partition 1

Download [DTD `dtd-r1.0.1`](https://www.robots.ox.ac.uk/~vgg/data/dtd/).
The official package includes images and ten sets of train/validation/test lists.
The model-merging convention uses the first partition:

| Purpose | Official lists | Images | Directory expected by the loader |
|---|---|---:|---|
| Training/validation pool | `train1.txt` + `val1.txt` | 3,760 | `dtd/train/<class>/` |
| Final test | `test1.txt` | 1,880 | `dtd/val/<class>/` |

Despite the directory name, `dtd/val` is the final test directory. `DTDVal`
creates an internal validation subset from `dtd/train` using the registry's
default 10% split and seed 0 (376 validation images for a 3,760-image pool).
Do not substitute the official `val1.txt` images for the final test.

For a fresh preparation from the official archive, run from the repository root:

```bash
conda activate ace-merging
pip install scipy pyarrow h5py requests beautifulsoup4
mkdir -p data/reproduction_assets data/datasets/vit_source
curl -L --fail \
  https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz \
  -o data/reproduction_assets/dtd-r1.0.1.tar.gz
tar -xzf data/reproduction_assets/dtd-r1.0.1.tar.gz \
  -C data/datasets/vit_source
python evaluation/vit/prepare_reproduction_data.py \
  --datasets DTD \
  --source data/datasets/vit_source \
  --target data/datasets/vit_reproduction \
  --assets data/reproduction_assets
```

The preparation helper arranges files under the target root and records image
lists and hashes in `preparation_manifest.json`. Use a fresh target when an
existing directory contains another partition. The source images must remain
available because preparation uses symbolic links.

## Other datasets in the eight-task suite

| Dataset | Preparation |
|---|---|
| MNIST, GTSRB, SVHN | Their torchvision-backed loaders download the required files when needed; retain the loader's expected root. |
| Stanford Cars | Prepare training/test images and the labeled `.mat` annotation files under `stanford_cars`. The original automatic-download URLs may fail; use a prepared copy with the official annotations. |
| SUN397 | Follow the upstream convention using `Training_01.txt` and `Testing_01.txt`. Arrange them as `sun397/train` and `sun397/val`; directory `val` is the final test set. |
| EuroSAT | Prepare `EuroSAT_splits/train`, `val`, and `test`, with 21,600/2,700/2,700 images. Matching counts alone does not establish matching image assignments. |
| RESISC45 | Prepare the images and the three `resisc45-*.txt` lists expected by the loader. |

The reproduction helpers `download_reproduction_assets.py` and
`prepare_reproduction_data.py` can prepare EuroSAT/RESISC45 and SUN397 from
available assets. SUN397's mirror option supplies a test-only public split.
Our mirror-based EuroSAT/SUN397 results have not been verified to use exactly the
same image assignments as the paper; record their manifests when reporting them.

## Additional tasks for 14 and 20 tasks

The 14-task suite adds PCAM, CIFAR100, STL10, OxfordIIITPet, Flowers102, and FER2013.
The 20-task suite also adds CIFAR10, Food101, RenderedSST2, EMNIST, FashionMNIST,
and KMNIST. The extended evaluation uses the EMNIST `digits` split.

`evaluation/vit/prepare_extended_vit_data.py` downloads the standard
torchvision-backed datasets. `prepare_extended_vit_special.py` downloads and
verifies PCAM, EMNIST, KMNIST, FER2013, and Food101 assets and prepares their test
files, including the FER2013/Food101 manifests. The full-test adapters are in
`extended_vit_data.py`. Every loader still needs its expected files under the
root passed to `--data-root`.

By default, the standard preparation helper creates `data/datasets/vit_extended`
and links its first eight datasets to `data/datasets/vit_reproduction`. Prepare
those eight datasets first, then pass `--data-root data/datasets/vit_extended`
for the extended suite. PCAM needs its official test image and label HDF5 files
under `PCAM/pcam`; EMNIST needs the `digits` files under `EMNIST`; FER2013 and
Food101 need their generated `test_manifest.json` and image directories. The
helpers prepare evaluation data; requesting validation may require additional
training files for the registry's validation split.

After preparing the first eight datasets, run:

```bash
python evaluation/vit/prepare_extended_vit_special.py \
  --root data/datasets/vit_extended
python evaluation/vit/prepare_extended_vit_data.py \
  --root data/datasets/vit_extended
```

Both helpers record preparation manifests. The special helper currently prepares
all of its listed datasets, even when only fourteen tasks will be evaluated.

## Evaluation paths and task selection

The explicit reproduction commands in [the reproduction guide](vit_reproduction.md)
take `--data-root`, `--model`, and `--task-count 8|14|20`. They evaluate the task
set recorded in the merge manifest, avoiding reliance on a manual task slice.

For a serialized `model.pt` produced by `main.py`, `evaluation/vit/eval_vit.py`
accepts `--merged-checkpoint`, `--data-root`, `--checkpoints`, `--model`, and
`--task-count 8|14|20`. It selects the complete task suite without editing the
source. Default paths are anchored to the repository, and relative paths passed
as arguments are resolved from the working directory. It evaluates test data by
default; add `--splits val test` only when every validation split is prepared.
Do not pass the explicit runner's `merged_state.pt` to this serialized-model
entry point.

For a loading error, check the resolved root, required annotations/split files,
image counts, and class-to-index order. If reporting an issue, include the dataset
name, resolved dataset-root path, and traceback. Checkpoint provenance is a
separate requirement: keep historical and corrected DTD weights labeled while
holding the test-image list fixed.
