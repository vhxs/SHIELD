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


def test_apply_conv2d_c1(check1) -> None:
    model = ConvLayer(check1.channel_size, check1.channel_size, check1.ker_size)
    model.eval()

    pt_conv = torch.squeeze(model(check1.rand_tensor), axis=0).detach().numpy()

    conv1 = check1.input_img.apply_conv(model.conv)
    dec_conv1 = conv1.decrypt_to_tensor(check1.cc, check1.keys).numpy().squeeze()

    assert np.allclose(dec_conv1, pt_conv, atol=1e-03), "Convolution result did not match between HE and PyTorch, failed image < shard"


def test_apply_conv2d_c2(check2) -> None:
    model = ConvLayer(check2.channel_size, check2.channel_size, check2.ker_size)
    model.eval()

    pt_conv = torch.squeeze(model(check2.rand_tensor), axis=0).detach().numpy()

    conv1 = check2.input_img.apply_conv(model.conv)
    dec_conv1 = conv1.decrypt_to_tensor(check2.cc, check2.keys).numpy().squeeze()

    assert np.allclose(dec_conv1, pt_conv, atol=1e-03), "Convolution result did not match between HE and PyTorch, failed channel < shard"


@pytest.mark.highmem
def test_apply_conv2d_c3(check3) -> None:
    model = ConvLayer(check3.channel_size, check3.channel_size, check3.ker_size)
    model.eval()

    pt_conv = torch.squeeze(model(check3.rand_tensor), axis=0).detach().numpy()

    conv1 = check3.input_img.apply_conv(model.conv)
    dec_conv1 = conv1.decrypt_to_tensor(check3.cc, check3.keys).numpy().squeeze()

    assert np.allclose(dec_conv1, pt_conv, atol=1e-03), "Convolution result did not match between HE and PyTorch, failed channel > shard"
