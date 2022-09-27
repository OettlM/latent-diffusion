import numpy as np
import albumentations as A

from os import listdir
from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2



class HER2Dataset(Dataset):
    def __init__(self, folder) -> None:
        super().__init__()

        self._folder = folder
        self._imgs = listdir(folder + "/images")
        self._labels = listdir(folder + "/labels")

        self.trans = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.ToFloat(), ToTensorV2()])

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, index):
        img = Image.open(self._folder + "/images/" + self._imgs[index])
        img = img.convert('RGB')
        label = Image.open(self._folder + "/labels/" + self._labels[index])

        applied = self.trans(image=np.array(img), mask=np.array(label))
        img = applied['image'].permute(1,2,0)
        label = applied['mask']

        out = {"image":img, "segmentation":label}
        return out