def test_is_dose_file():
    from data.dose_folder import is_dose_file

    assert is_dose_file("/home/sliberman/data/processed_V5/train/original/dose_100.dcm") is True
    assert is_dose_file("/home/sliberman/data/processed_V5/train/original/density_100.npy") is False


def test_is_density_file():
    from data.dose_folder import is_density_file

    assert is_density_file("/home/sliberman/data/processed_V5/train/original/dose_100.dcm") is False
    assert is_density_file("/home/sliberman/data/processed_V5/train/original/density_100.npy") is True

    
def test_make_dataset():
    from data.dose_folder import make_dataset
    
    dir = "/home/sliberman/data/processed_V5/train/original"
    dataset = make_dataset(dir)
    assert len(dataset) == 1959


class Options:
        dataroot = "/home/sliberman/data/processed_V5/"
        phase = "train"
        max_dataset_size = float("inf")
        input_nc = 2
        output_nc = 1
        direction = "AtoB"
    

def test_dataset():
    from data.aligned_dose_dataset import AlignedDataset

    opt = Options()

    ds = AlignedDataset(opt)

    assert all([k in ds[0].keys() for k in ["A", "B", "density"]])
