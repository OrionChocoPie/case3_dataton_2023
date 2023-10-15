import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class CarsDataset(Dataset):
    def __init__(self, image_pathes, image_classes, indexes, class2index, transform, length=None):
        self.image_pathes = image_pathes[indexes]
        self.image_classes = image_classes[indexes]
        self.class2index = class2index
        self.num_classes = len(class2index)
        self.transform = transform
        self.length = length

    def __len__(self):
        if self.length is None:
            return len(self.image_pathes)
        else:
            return self.length

    def _get_sample(self, index):
        image_path = self.image_pathes[index]
        label = self.image_classes[index]
        image = Image.open(image_path)

        return image, self.class2index[label]

    def __getitem__(self, index):
        if self.length is not None:
            index = np.random.randint(len(self.images))
        image, label = self._get_sample(index)

        image = self.transform(image)

        return image, label
