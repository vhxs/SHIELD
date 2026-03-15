# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import torch
import numpy as np


class LinearLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.linear_one = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear_one(x)
        return x


def test_apply_linear_c1(check1) -> None:
    linear = LinearLayer(len(check1.rand_tensor.flatten()), check1.rand_tensor.shape[0])
    linear.eval()

    pt_linear = linear(check1.rand_tensor.flatten()).detach().numpy()
    he_linear = check1.input_img.apply_linear(linear.linear_one)
    dec_linear = check1.cc.decrypt(check1.keys.secretKey, he_linear)[0:check1.rand_tensor.shape[0]]

    assert np.allclose(dec_linear, pt_linear, atol=1e-03), "Linear result did not match between HE and PyTorch, failed image < shard"


def test_apply_linear_c2(check2) -> None:
    linear = LinearLayer(len(check2.rand_tensor.flatten()), check2.rand_tensor.shape[0])
    linear.eval()

    pt_linear = linear(check2.rand_tensor.flatten()).detach().numpy()
    he_linear = check2.input_img.apply_linear(linear.linear_one)
    dec_linear = check2.cc.decrypt(check2.keys.secretKey, he_linear)[0:check2.rand_tensor.shape[0]]

    assert np.allclose(dec_linear, pt_linear, atol=1e-03), "Linear result did not match between HE and PyTorch, failed channel < shard"
