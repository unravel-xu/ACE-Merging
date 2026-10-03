import pathlib
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

class PytorchStanfordCars(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError(
                "Scipy is not found. Please install it with: pip install scipy"
            )

        root = pathlib.Path(root).expanduser()
        super().__init__(str(root), transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))

        self._base_folder = root / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            # Prepared releases place the labeled test annotations in either location.
            candidates = (
                devkit / "cars_test_annos_withlabels.mat",
                self._base_folder / "cars_test_annos_withlabels.mat",
            )
            self._annotations_mat_path = next(
                (path for path in candidates if path.is_file()), candidates[0]
            )
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                f"Stanford Cars dataset not found or incomplete.\n"
                f"Expected base folder: {self._base_folder}\n"
                f"Expected image folder: {self._images_base_path}\n"
                f"Expected annotation file: {self._annotations_mat_path}\n"
                f"Need train annotations: {devkit / 'cars_train_annos.mat'}\n"
                f"Test annotations may also be placed at: {self._base_folder / 'cars_test_annos_withlabels.mat'}\n"
                f"Need class metadata: {devkit / 'cars_meta.mat'}"
            )

        annotations = sio.loadmat(
            str(self._annotations_mat_path),
            squeeze_me=True
        )["annotations"]

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                int(annotation["class"]) - 1,
            )
            for annotation in annotations
        ]

        self.classes = sio.loadmat(
            str(devkit / "cars_meta.mat"),
            squeeze_me=True
        )["class_names"].tolist()

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_path, target = self._samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def download(self) -> None:
        if self._check_exists():
            return

        raise RuntimeError(
            "Automatic download for Stanford Cars is disabled because the original links are unavailable. "
            f"Please manually place the dataset under: {self._base_folder}"
        )

    def _check_exists(self) -> bool:
        devkit = self._base_folder / "devkit"

        if not devkit.is_dir():
            return False

        if not (devkit / "cars_meta.mat").exists():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()

class Cars:
    def __init__(
        self,
        preprocess,
        location="ACE-Merging/data/datasets/vit",
        batch_size=32,
        num_workers=16,
    ):

        self.train_dataset = PytorchStanfordCars(
            location,
            split="train",
            transform=preprocess,
            download=False,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = PytorchStanfordCars(
            location,
            split="test",
            transform=preprocess,
            download=False,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        idx_to_class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        self.classnames = [
            idx_to_class[i].replace("_", " ")
            for i in range(len(idx_to_class))
        ]
