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


class DatasetCells(Dataset):
    def __init__(self, folder, cluster, aug) -> None:
        super().__init__()

        if cluster:
            local_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
            folder_name = os.path.basename(folder)

            target_dir = local_dir
            copy_tree(folder, target_dir)

            self._folder = target_dir
        else:
            self._folder = folder

        self._cell_types = ["Eosinophile", "Erythrozyt", "Lymohozyten", "Makrophagen", "Mastzellen", "Mehrkernige", "Neutrophile"]

        self._files = []
        for cell_type in self._cell_types:
            cell_files = listdir(self._folder + "/" + cell_type)
            self._files.append(cell_files)       

        self._num_imgs = 20000

        if aug:
            self.trans = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.Resize(64,64),
                                    A.ToFloat(), ToTensorV2()])
        else:
            self.trans = A.Compose([A.Resize(64,64), A.ToFloat(), ToTensorV2()])

    def __len__(self):
        return self._num_imgs

    def __getitem__(self, index):
        cell_int = np.random.randint(0,7)
        cell_list = self._files[cell_int]
        file_int = np.random.randint(0,len(cell_list))
        cell_file = cell_list[file_int]

        img = Image.open(self._folder + "/" + self._cell_types[cell_int] + "/" + cell_file)
        img = np.array(img.convert('RGB'))

        applied = self.trans(image=img)
        img = applied['image'].permute(1,2,0)

        out = {"image":img*2-1, "class_label":cell_int, "human_label":self._cell_types[cell_int]}
        return out