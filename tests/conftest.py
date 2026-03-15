# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import pytest
import torch
import numpy as np

from shield.cnn_context import create_cnn_context
from shield.he_cnn.utils import *

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class Info():
    def __init__(self, mult_depth=30, scale_factor_bits=40, batch_size=32 * 32 * 32, max=255, min=0, h=128, w=128, channel_size=3, ker_size=3):
        self.mult_depth = mult_depth
        self.scale_factor_bits = scale_factor_bits
        self.batch_size = batch_size
        self.max = max
        self.min = min
        self.h = h
        self.w = w
        self.channel_size = channel_size
        self.ker_size = ker_size

        rand_tensor = (max - min) * torch.rand((channel_size, h, w)) + min
        self.rand_tensor = rand_tensor

        self.cc, self.keys = create_cc_and_keys(batch_size, mult_depth=mult_depth, scale_factor_bits=scale_factor_bits, bootstrapping=False)
        self.input_img = create_cnn_context(self.rand_tensor, self.cc, self.keys.publicKey, verbose=True)


@pytest.fixture(scope="session")
def check1():
    return Info(30, 40, 32 * 32 * 32, 1, -1, 64, 64, 4, 3)


@pytest.fixture(scope="session")
def check2():
    return Info(30, 40, 32 * 32 * 32, 1, -1, 64, 64, 1, 3)


@pytest.fixture(scope="session")
def check3():
    return Info(30, 40, 32, 1, -1, 16, 16, 2, 3)
