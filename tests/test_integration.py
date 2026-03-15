# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import os
import pytest
import numpy as np
import torch

from palisade_he_cnn.cnn_context import create_cnn_context
from palisade_he_cnn.he_cnn.utils import create_cc_and_keys
from palisade_he_cnn.small_model import SmallModel, train_small_model

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
WEIGHTS_PATH = os.path.join(FIXTURES_DIR, "small_model.pt")
INPUT_PATH = os.path.join(FIXTURES_DIR, "small_model_input.pt")


@pytest.mark.integration
def test_small_model_inference():
    # Load or train model
    model = SmallModel(activation='gelu', pool_method='avg')
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH))
    else:
        train_small_model(output_path=WEIGHTS_PATH)
        model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval()

    # Load fixed input
    fixture = torch.load(INPUT_PATH)
    x = fixture['x']

    # Set up HE context
    mult_depth = 30
    scale_factor_bits = 40
    batch_size = 32 * 32 * 32
    cc, keys = create_cc_and_keys(batch_size, mult_depth=mult_depth,
                                  scale_factor_bits=scale_factor_bits,
                                  bootstrapping=False)

    # HE inference
    input_img = create_cnn_context(x, cc, keys.publicKey, verbose=False)

    layer = model.model_layers.conv1
    cnn = input_img.apply_conv(layer[0], layer[1])
    cnn = cnn.apply_gelu()
    cnn = cnn.apply_pool()

    layer = model.model_layers.conv2
    cnn = cnn.apply_conv(layer[0], layer[1])
    cnn = cnn.apply_gelu()
    cnn = cnn.apply_pool()

    layer = model.model_layers.conv3
    cnn = cnn.apply_conv(layer[0], layer[1])
    cnn = cnn.apply_gelu()

    layer = model.model_layers.classifier[1]
    logits_he = cc.decrypt(keys.secretKey, cnn.apply_linear(layer))[:10]

    # Plaintext inference
    logits_pt = model(x.unsqueeze(0)).detach().numpy().ravel()

    assert np.allclose(logits_he, logits_pt, atol=1e-3), (
        f"HE logits do not match plaintext logits.\n"
        f"HE:        {logits_he}\n"
        f"Plaintext: {logits_pt}"
    )
