import random
import torch


def random_crop(data, output_size):
    original = data["A"]
    original_variance = data["A_var"]
    original_density = data["density"]

    target = data["B"]
    target_variance = data["B_var"]
    target_density = data["density"]

    assert all(i >= j for i, j in zip(original.shape[-3:], output_size))
    assert all(i == j for i, j in zip(original.shape[-3:], target.shape[-3:]))

    orig_x, orig_y, orig_z = original.shape[-3:]
    out_x, out_y, out_z = output_size

    new_original_shape = list(original.shape[:-3]) + [out_x, out_y, out_z]
    new_target_shape = list(target.shape[:-3]) + [out_x, out_y, out_z]

    new_original_dose = torch.empty(new_original_shape)
    new_original_variance = torch.empty(new_original_shape)
    new_original_density = torch.empty(new_original_shape)
    new_target_dose = torch.empty(new_target_shape)
    new_target_variance = torch.empty(new_target_shape)
    new_target_density = torch.empty(new_target_shape)

    # for each sample in the batch do
    for i, (o_sample, ov_sample, od_sample, t_sample, tv_sample, td_sample) in enumerate(zip(original,
                                                                                             original_variance,
                                                                                             original_density,
                                                                                             target,
                                                                                             target_variance,
                                                                                             target_density)):
        # sample crop starting point
        x = random.randint(0, orig_x - out_x)
        y = random.randint(0, orig_y - out_y)
        z = random.randint(0, orig_z - out_z)

        # slice cubes
        o_sample = o_sample[..., x:x + out_x, y:y + out_y, z:z + out_z]
        od_sample = od_sample[..., x:x + out_x, y:y + out_y, z:z + out_z]
        ov_sample = ov_sample[..., x:x + out_x, y:y + out_y, z:z + out_z]
        t_sample = t_sample[..., x:x + out_x, y:y + out_y, z:z + out_z]
        td_sample = td_sample[..., x:x + out_x, y:y + out_y, z:z + out_z]
        tv_sample = tv_sample[..., x:x + out_x, y:y + out_y, z:z + out_z]

        new_original_dose[i] = o_sample
        new_original_density[i] = od_sample
        new_original_variance[i] = ov_sample
        new_target_dose[i] = t_sample
        new_target_density[i] = td_sample
        new_target_variance[i] = tv_sample

    return {
        "A": new_original_dose,
        "A_var": new_original_variance,
        "density": new_original_density,
        "B": new_target_dose,
        "B_var": new_target_variance,
        "A_paths": data["A_paths"],
        "B_paths": data["B_paths"]
    }


def random_flip(data):
    """Randomly flip the doses and densities, except for the Y axis which is where the magnetic field is"""
    original = data["A"]
    original_variance = data["A_var"]
    original_density = data["density"]

    target = data["B"]
    target_variance = data["B_var"]
    target_density = data["density"]

    # for each sample in the batch do
    for i, (o_sample, ov_sample, od_sample, t_sample, tv_sample, td_sample) in enumerate(zip(original,
                                                                                             original_variance,
                                                                                             original_density,
                                                                                             target,
                                                                                             target_variance,
                                                                                             target_density)):
        # sample crop starting point
        flip_x = random.randint(0, 1)
        flip_z = random.randint(0, 1)

        # If 1, flip. Else, remain
        x = -1 if flip_x == 1 else 1
        z = -1 if flip_z == 1 else 1

        # slice cubes
        o_sample[:] = torch.from_numpy(o_sample.numpy()[..., ::x, :, ::z].copy())
        od_sample[:] = torch.from_numpy(od_sample.numpy()[..., ::x, :, ::z].copy())
        ov_sample[:] = torch.from_numpy(ov_sample.numpy()[..., ::x, :, ::z].copy())
        t_sample[:] = torch.from_numpy(t_sample.numpy()[..., ::x, :, ::z].copy())
        td_sample[:] = torch.from_numpy(td_sample.numpy()[..., ::x, :, ::z].copy())
        tv_sample[:] = torch.from_numpy(tv_sample.numpy()[..., ::x, :, ::z].copy())

    return data
