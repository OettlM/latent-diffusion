from distutils.dir_util import copy_tree
import os
import torch
import numpy as np
import albumentations as A

from os import listdir
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2



class HER2Dataset(Dataset):
    def __init__(self, folder, cluster, target, aug) -> None:
        super().__init__()

        if cluster:
            local_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
            folder_name = os.path.basename(folder)

            target_dir = os.path.join(local_dir, folder_name)
            copy_tree(folder, target_dir)

            self._folder = target_dir
        else:
            self._folder = folder

        self._target = target
        
        if self._target == "label_syn":
            self._img_target = "img_syn"
        else:
            self._img_target = "img"

        imgs = listdir(self._folder + "/" + self._img_target)
        self._num_imgs = len(imgs)

        if aug:
            self.trans = A.Compose([A.HorizontalFlip(), A.VerticalFlip(),
                                    A.ToFloat(), ToTensorV2()])
        else:
            self.trans = A.Compose([A.ToFloat(), ToTensorV2()])

    def __len__(self):
        return self._num_imgs

    def __getitem__(self, index):
        img = Image.open(self._folder + "/" +self._img_target + "/" + f"{str(index).zfill(4)}.png")
        img = np.array(img.convert('RGB'))
        label = np.array(Image.open(self._folder + "/"+self._target+"/" + f"{str(index).zfill(4)}.png"))

        applied = self.trans(image=img, mask=label)
        img = applied['image'].permute(1,2,0)
        label = applied['mask']
        seg = one_hot(label.to(torch.int64), num_classes=6)

        out = {"image":img*2-1, "segmentation":seg, "label":label, "index":index}
        return out