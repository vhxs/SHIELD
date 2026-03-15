# SHIELD: Secure Homomorphic Inference for Encrypted Learning on Data

SHIELD is a library for evaluating pre-trained convolutional neural networks on homomorphically encrypted images. It includes code for training models that are suitable for homomorphic evaluation. Implemented neural network operations include convolution, average pooling, GELU, and linear layers.

This code was used to run the experiments supporting the following paper: [High-Resolution Convolutional Neural Networks on Homomorphically Encrypted Data via Sharding Ciphertexts](https://arxiv.org/abs/2306.09189). However, operators defined in this project are generic enough to build arbitrary convolutional neural networks as specified in the paper.

## Requirements

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Python 3.12 is required.

OpenFHE Python bindings are provided via [SHIELD-core](https://github.com/vhxs/SHIELD-core) and are installed automatically as part of `uv sync`. Pre-built wheels are available for Linux (x86_64) and macOS (arm64).

To install:

```
uv sync
```

For running unit tests and the small neural network, 32GB of RAM is recommended. For hardware requirements needed to reproduce results for the larger ResNet architectures, see the paper for details.

## Features

SHIELD implements the following neural network operators:

- Convolution
- Average pooling
- Batch normalization (fused with convolution operators for performance)
- Linear
- GELU (Gaussian Error Linear Unit, a smooth alternative to ReLU)
- Upsample

For performance reasons, the core of these algorithms are implemented in C++ via the SHIELD-core companion project, with this project providing a user-friendly Python interface.

The following neural network architectures are implemented: a three-layer convolutional neural network (used for integration testing), and variations on ResNet including ResNet9 and ResNet50. Training code includes kurtosis regularization required for homomorphic inference. See the referenced paper for more details on the algorithms and performance metrics.

## Running the code

### Unit tests

Tests are run with `pytest`. The default test run excludes tests requiring more than 7GB of RAM:

```
uv run pytest tests/ -m "not integration and not highmem"
```

To run high-memory tests (requires ~16GB RAM):

```
uv run pytest tests/ -m "highmem"
```

To run the end-to-end integration test:

```
uv run pytest tests/ -m "integration"
```

### A small neural network

`src/shield/small_model.py` defines a 3-layer convolutional neural network and includes training code. To train on MNIST:

```
uv run python -c "from shield.small_model import train_small_model; train_small_model()"
```

This saves model weights to `small_model.pt`. Example weights are included in `tests/fixtures/`. To run homomorphic inference:

```
uv run python examples/small_model_inference.py
```

### Larger neural networks

Scripts to train larger models are in `training/`. Scripts to run inference with these models are in `inference/`. Due to the significant resources required, weights for the larger ResNet architectures will be added to this repository in the future.

## Citation and Acknowledgements

Please cite this work as follows:

```
@misc{maloney2024highresolutionconvolutionalneuralnetworks,
      title={High-Resolution Convolutional Neural Networks on Homomorphically Encrypted Data via Sharding Ciphertexts},
      author={Vivian Maloney and Richard F. Obrecht and Vikram Saraph and Prathibha Rama and Kate Tallaksen},
      year={2024},
      eprint={2306.09189},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2306.09189},
}
```

In addition to the authors on the supporting manuscript (Vivian Maloney, Freddy Obrecht, Vikram Saraph, Prathibha Rama, and Kate Tallaksen), Lindsay Spriggs and Court Climer also contributed to this work by testing the software and integrating it with internal infrastructure.
