import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from src.datasets.cars import PytorchStanfordCars


class StanfordCarsTests(unittest.TestCase):
    def test_labeled_test_annotations_in_either_supported_location(self):
        for location in ('devkit', '.'):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / 'stanford_cars'
                (base / 'devkit').mkdir(parents=True)
                (base / 'cars_test').mkdir()
                annotations = np.empty((1, 2), dtype=[('fname', object), ('class', object)])
                for index in range(2):
                    filename = f'{index + 1:05d}.jpg'
                    Image.new('RGB', (3, 3), color=(index * 100, 20, 30)).save(base / 'cars_test' / filename)
                    annotations[0, index] = (filename, index + 1)
                annotation_path = base / location / 'cars_test_annos_withlabels.mat'
                savemat(annotation_path, {'annotations': annotations})
                savemat(base / 'devkit/cars_meta.mat', {'class_names': np.array(['car_a', 'car_b'], dtype=object)})

                dataset = PytorchStanfordCars(directory, split='test')
                self.assertEqual(dataset._annotations_mat_path, annotation_path)
                self.assertEqual(len(dataset), 2)
                self.assertEqual(dataset.class_to_idx, {'car_a': 0, 'car_b': 1})
                for index in range(2):
                    image, label = dataset[index]
                    self.assertEqual(image.mode, 'RGB')
                    self.assertEqual(label, index)


if __name__ == '__main__':
    unittest.main()
