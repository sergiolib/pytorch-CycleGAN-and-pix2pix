import os
import itertools
import pathlib


DOSE_EXTENSIONS = ["dcm"]
DOSE_PREFIX = ["dose"]
DENSITY_EXTENSIONS = ["npy"]
DENSITY_PREFIX = ["density"]

def make_dataset(dir, max_dataset_size=float("inf")):
    doses = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_dose_file(fname):
                path = os.path.join(root, fname)
                doses.append(path)
    return doses[:min(max_dataset_size, len(doses))]


def make_density_dataset(dir, max_dataset_size=float("inf")):
    densities = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_density_file(fname):
                path = os.path.join(root, fname)
                densities.append(path)
    return densities[:min(max_dataset_size, len(densities))]


def is_dose_file(filename):
    filename = pathlib.Path(filename)
    filename = filename.name
    return any(filename.endswith(extension) and filename.startswith(prefix)
               for extension, prefix in itertools.product(DOSE_EXTENSIONS, DOSE_PREFIX))


def is_density_file(filename):
    filename = pathlib.Path(filename)
    filename = filename.name
    return any(filename.endswith(extension) and filename.startswith(prefix)
               for extension, prefix in itertools.product(DENSITY_EXTENSIONS, DENSITY_PREFIX))
