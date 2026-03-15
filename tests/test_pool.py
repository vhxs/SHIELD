# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import pytest
import torch
import numpy as np


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

    def forward(self, x):
        x = self.conv(x)
        return x


def test_apply_pool_c1(check1) -> None:
    model = ConvLayer(check1.channel_size, check1.channel_size, check1.ker_size)
    model.eval()

    pt_pool = torch.nn.AvgPool2d(2)(model(check1.rand_tensor)).detach().numpy()

    conv1 = check1.input_img.apply_conv(model.conv)
    dec_pool = conv1.apply_pool().decrypt_to_tensor(check1.cc, check1.keys).numpy()

    assert np.allclose(dec_pool, pt_pool, atol=1e-03), "Pooling result did not match between HE and PyTorch, failed image < shard"


def test_apply_pool_c2(check2) -> None:
    model = ConvLayer(check2.channel_size, check2.channel_size, check2.ker_size)
    model.eval()

    pt_pool = torch.nn.AvgPool2d(2)(model(check2.rand_tensor)).detach().numpy()

    conv1 = check2.input_img.apply_conv(model.conv)
    dec_pool = conv1.apply_pool().decrypt_to_tensor(check2.cc, check2.keys).numpy()

    assert np.allclose(dec_pool, pt_pool, atol=1e-03), "Pooling result did not match between HE and PyTorch, failed channel < shard"


@pytest.mark.highmem
def test_apply_pool_c3(check3) -> None:
    model = ConvLayer(check3.channel_size, check3.channel_size, check3.ker_size)
    model.eval()

    pt_pool = torch.nn.AvgPool2d(2)(model(check3.rand_tensor)).detach().numpy()

    conv1 = check3.input_img.apply_conv(model.conv)
    dec_pool = conv1.apply_pool().decrypt_to_tensor(check3.cc, check3.keys).numpy()

    assert np.allclose(dec_pool, pt_pool, atol=1e-03), "Pooling result did not match between HE and PyTorch, failed channel > shard"
