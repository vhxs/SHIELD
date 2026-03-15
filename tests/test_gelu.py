# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import torch
import numpy as np


def test_apply_gelu_c1(check1) -> None:
    pt_gelu = torch.nn.GELU()(check1.rand_tensor)
    dec_gelu = check1.input_img.apply_gelu().decrypt_to_tensor(check1.cc, check1.keys).numpy()

    assert np.allclose(dec_gelu, pt_gelu, atol=1e-03), "GELU result did not match between HE and PyTorch, failed image < shard"


def test_apply_gelu_c2(check2) -> None:
    pt_gelu = torch.nn.GELU()(check2.rand_tensor)
    dec_gelu = check2.input_img.apply_gelu().decrypt_to_tensor(check2.cc, check2.keys).numpy()

    assert np.allclose(dec_gelu, pt_gelu, atol=1e-03), "GELU result did not match between HE and PyTorch, failed channel < shard"


def test_apply_gelu_c3(check3) -> None:
    pt_gelu = torch.nn.GELU()(check3.rand_tensor)
    dec_gelu = check3.input_img.apply_gelu().decrypt_to_tensor(check3.cc, check3.keys).numpy()

    assert np.allclose(dec_gelu, pt_gelu, atol=1e-03), "GELU result did not match between HE and PyTorch, failed channel > shard"
