import pytest
import torch

from models.models import FCN


@pytest.fixture
def fcn() -> FCN:
    return FCN()


def test_fcn_forward_pass_cpu(fcn: FCN) -> None:
    input_tensor = torch.randn(32, 1, 28, 28)
    output = fcn.forward(input_tensor)
    assert output.shape == (32, 10)


def test_fcn_forward_pass_gpu(fcn: FCN) -> None:
    input_tensor = torch.randn(32, 1, 28, 28).cuda()
    fcn = fcn.to("cuda")
    output = fcn.forward(input_tensor)
    assert output.shape == (32, 10)


def test_fcn_initialization(fcn: FCN) -> None:
    assert isinstance(fcn, FCN)
    assert len(fcn.network) == 3


def test_fcn_num_layers() -> None:
    try:
        _ = FCN(num_layers=1)
    except ValueError as e:
        assert str(e) == "Number of layers must be at least 2, got 1 number of layers"
