import os.path

import h5py
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.dose_folder import make_dataset, make_density_dataset, make_variance_dataset
from pydicom import dcmread


class AlignedDoseH5Dataset(BaseDataset):
    """A dataset class for paired doses dataset.

    It assumes that the directory '/path/to/data/train' contains two directories: original and target, and that doses in original are A, while those in target are B.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataroot = self.opt.dataroot
        self.mode = self.opt.phase
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        with h5py.File(self.dataroot, "r") as f:
            original_dose = f["original"][self.mode][f"dose_{index}"][...]
            original_variance = f["original"][self.mode][f"variance_{index}"][...]

            target_dose = f["target"][self.mode][f"dose_{index}"][...]
            target_variance = f["target"][self.mode][f"variance_{index}"][...]

            density = f["density"][...]

        loaded = {
            "A": original_dose.astype(np.float32),
            "A_var": original_variance.astype(np.float32),
            "B": target_dose.astype(np.float32),
            "B_var": density.astype(np.float32),
            "variance": target_variance.astype(np.float32),
            "density": density.astype(np.float32),
        }

        return loaded

    def __len__(self):
        """Return the total number of images in the dataset."""
        with h5py.File(self.dataroot, "r") as f:
            mode_files = f["original"][self.mode]
            return len([i for i in mode_files.keys() if "dose" in i])
