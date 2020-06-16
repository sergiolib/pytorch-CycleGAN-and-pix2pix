import math
import numpy as np
import sklearn.preprocessing as pp
import torch
import torch.nn as nn


class PowerTransformModel(nn.Module):
    def __init__(self, load_from: pp.PowerTransformer = None):
        super().__init__()
        self.params = nn.Parameter(torch.ones(6), requires_grad=False)
        self.param_names = ["Lambda dose", "Lambda density", "Mean dose",
                            "Mean density", "Log scale dose", "Log scale density"]
        if load_from is None:
            self.params.requires_grad = True
        else:
            self.params[0] = load_from.lambdas_[0]
            self.params[1] = load_from.lambdas_[1]
            self.params[2] = load_from._scaler.mean_[0]
            self.params[3] = load_from._scaler.mean_[1]
            self.params[4] = math.log(load_from._scaler.scale_[0])
            self.params[5] = math.log(load_from._scaler.scale_[1])

    def forward(self, x):
        x = x.clone()

        batch_size = x.shape[0]
        channels = x.shape[1]
        reshaped_x = x.view(batch_size, channels, -1)

        # when reshaped_x >= 0
        for sample in range(batch_size):
            for col, lmbda in enumerate(self.params):
                if reshaped_x.shape[1] <= col:
                    break
                pos = reshaped_x[sample, col] >= 0

                if abs(lmbda) < np.spacing(1.):
                    reshaped_x[sample, col, pos] = torch.exp(reshaped_x[sample, col, pos]) - 1
                else:  # lmbda != 0
                    reshaped_x[sample, col, pos] = (torch.pow(reshaped_x[sample, col, pos] + 1, lmbda) - 1) / lmbda

                # when x < 0
                if abs(lmbda - 2) > np.spacing(1.):
                    reshaped_x[sample, col, ~pos] = -torch.pow(-reshaped_x[sample, col, ~pos] + 1, 2 - lmbda) / (2 - lmbda)
                else:  # lmbda == 2
                    reshaped_x[sample, col, ~pos] = -torch.log1p(-reshaped_x[sample, col, ~pos])

                # Standarize
                reshaped_x[sample, col] -= self.params[2 + col]
                reshaped_x[sample, col] /= torch.exp(self.params[4 + col])

        return x

    def inverse_transform(self, x):
        x = x.clone()

        batch_size = x.shape[0]
        channels = x.shape[1]
        reshaped_x = x.view(batch_size, channels, -1)


        # when reshaped_x >= 0
        for sample in range(batch_size):
            for col, lmbda in enumerate(self.lambdas):
                # Unstandarize
                reshaped_x[sample, col] *= torch.exp(self.logscale[col])
                reshaped_x[sample, col] += self.mean[col]
                if reshaped_x.shape[1] <= col:
                    break
                pos = reshaped_x[sample, col] >= 0

                if abs(lmbda) < np.spacing(1.):
                    reshaped_x[sample, col, pos] = torch.exp(reshaped_x[sample, col, pos]) - 1
                else:  # lmbda != 0
                    reshaped_x[sample, col, pos] = torch.pow(reshaped_x[sample, col, pos] * lmbda + 1, 1.0 / lmbda) - 1

                # when x < 0
                if abs(lmbda - 2) > np.spacing(1.):
                    reshaped_x[sample, col, ~pos] = 1 - torch.pow(-(2 - lmbda) * reshaped_x[sample, col, ~pos] + 1,
                                                                  1 / (2 - lmbda))
                else:  # lmbda == 2
                    reshaped_x[sample, col, ~pos] = 1 - torch.exp(-reshaped_x[sample, col, ~pos])

        return x


class ZCAModel(nn.Module):
    def __init__(self, load_from: dict = None):
        super().__init__()
        initial = torch.zeros(4)
        initial[0] = 0.05
        initial[1] = 5
        initial[2] = math.log(2)
        initial[3] = math.log(8)
        self.params = nn.Parameter(initial, requires_grad=False)
        self.param_names = ["Mean dose", "Mean density", "Log scale dose", "Log scale density"]

        if load_from is None:
            self.params.requires_grad = True
        else:
            self.params[0] = load_from["mean_dose"]
            self.params[1] = load_from["mean_density"]
            self.params[2] = math.log(load_from["std_dose"])
            self.params[3] = math.log(load_from["std_density"])

    def forward(self, x):
        x = x.clone()

        batch_size = x.shape[0]
        channels = x.shape[1]
        reshaped_x = x.view(batch_size, channels, -1)

        for sample in range(batch_size):
            for col in range(reshaped_x.shape[1]):
                # Standarize
                reshaped_x[sample, col] -= self.params[col]
                reshaped_x[sample, col] /= torch.exp(self.params[2 + col])

        return x

    def inverse_transform(self, x):
        x = x.clone()

        batch_size = x.shape[0]
        channels = x.shape[1]
        reshaped_x = x.view(batch_size, channels, -1)


        # when reshaped_x >= 0
        for sample in range(batch_size):
            for col in range(reshaped_x.shape[1]):
                # Unstandarize
                reshaped_x[sample, col] *= torch.exp(self.params[2 + col])
                reshaped_x[sample, col] += self.params[col]

        return x


class LinearScalerModel(nn.Module):
    def __init__(self, load_from: dict = None):
        super().__init__()
        initial = torch.zeros(4)
        initial[2] = 1
        initial[3] = 10
        self.params = nn.Parameter(initial, requires_grad=False)
        self.param_names = ["Min dose", "Min density", "Max dose", "Max density"]

        if load_from is None:
            self.params.requires_grad = True
        else:
            self.params[0] = load_from["min_dose"]
            self.params[1] = load_from["min_density"]
            self.params[2] = load_from["max_dose"]
            self.params[3] = load_from["max_density"]

    def forward(self, x):
        x = x.clone()

        x[:, 0] -= self.params[0]
        x[:, 0] /= (self.params[2] - self.params[0])

        if x.shape[1] == 2:
            x[:, 1] -= self.params[1]
            x[:, 1] /= (self.params[3] - self.params[1])

        return x


class IdentityScaler:
    def __call__(self, x):
        return x
