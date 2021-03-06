import os.path
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.dose_folder import make_dataset, make_density_dataset, make_variance_dataset
from pydicom import dcmread


class AlignedDoseDataset(BaseDataset):
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
        self.dir_A = os.path.join(opt.dataroot, opt.phase,
                                  "original")  # get the doses directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase,
                                  "target")  # get the doses directory
        self.A_paths = sorted(make_dataset(
            self.dir_A, opt.max_dataset_size))  # get dose paths
        self.B_paths = sorted(make_dataset(
            self.dir_B, opt.max_dataset_size))  # get dose paths
        self.A_density_paths = sorted(
            make_density_dataset(self.dir_A,
                                 opt.max_dataset_size))  # get density paths
        self.B_density_paths = sorted(
            make_density_dataset(self.dir_B,
                                 opt.max_dataset_size))  # get density paths
        self.A_var_paths = sorted(
            make_variance_dataset(self.dir_A,
                                  opt.max_dataset_size))  # get dose paths
        self.B_var_paths = sorted(
            make_variance_dataset(self.dir_B,
                                  opt.max_dataset_size))  # get dose paths
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
        # Doses
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_readfile = dcmread(str(A_path))
        B_readfile = dcmread(str(B_path))

        A_readfile.fix_meta_info()
        B_readfile.fix_meta_info()
        
        A = A_readfile.pixel_array * A_readfile.DoseGridScaling
        B = B_readfile.pixel_array * B_readfile.DoseGridScaling

        # Variances
        A_var_path = self.A_var_paths[index]
        B_var_path = self.B_var_paths[index]
        A_var_readfile = dcmread(str(A_var_path))
        B_var_readfile = dcmread(str(B_var_path))

        A_var_readfile.fix_meta_info()
        B_var_readfile.fix_meta_info()
        
        A_var = A_var_readfile.pixel_array * A_var_readfile.DoseGridScaling
        B_var = B_var_readfile.pixel_array * B_var_readfile.DoseGridScaling
        
        # Density
        density = np.load(self.A_density_paths[index])

        return {
            'A': A.astype(np.float32),
            'B': B.astype(np.float32),
            'A_var': A_var.astype(np.float32),
            'B_var': B_var.astype(np.float32),
            'density': density.astype(np.float32),
            'A_paths': A_path,
            'B_paths': B_path
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
