from __future__ import annotations
import math

import torch
import torch.nn as nn
from torch.jit._script import script


@script
def ripple_linear_func(
    input: torch.Tensor,
    out_features: int,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:

    # Register output sizes
    input_size = input.size()
    output_size = list(input_size)
    output_size[-1] = out_features

    # flatten the input
    flattened_input = input.flatten(end_dim=-2)

    if bias is not None:
        operation_result = (
            torch.einsum(
                "io,bio->bo",
                weight[:, :, 0],
                torch.sin(
                    torch.einsum("bi,io->bio", flattened_input, weight[:, :, 1])
                    + bias[1:, :]
                ),
            )
            + bias[0, :]
        )
    else:
        operation_result = torch.einsum(
            "io,bio->bo",
            weight[:, :, 0],
            torch.sin(torch.einsum("bi,io->bio", flattened_input, weight[:, :, 1])),
        )

    return operation_result.view(output_size)


class RippleLinear(nn.Module):
    """
    A simple trigonometric linear layer composed of trigonometric neurons; experimental
    neuron type to avoid segmenting the classification field to piece-wise linear segments.
    Should work exactly like the regular input layer, but this time we have ~3n parameters with biases
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initializes the RippleNet module.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            device (str or torch.device, optional): Device on which to allocate the tensors. Defaults to None.
            dtype (torch.dtype, optional): Data type of the weight and bias tensors. Defaults to None.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((in_features, out_features, 2), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((in_features + 1, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the layer.

        This method initializes the weight and bias parameters of the layer using the Kaiming uniform initialization
        method. The weight parameter is initialized with a Kaiming uniform distribution with a=math.sqrt(5), which is
        equivalent to initializing with uniform(-1/sqrt(in_features), 1/sqrt(in_features)). The bias parameter is
        initialized with a uniform distribution between -bound and bound, where bound is 1 / math.sqrt(fan_in) if
        fan_in > 0, otherwise it is set to 0.

        Returns:
            None
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the RippleNet module.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return ripple_linear_func(input, self.out_features, self.weight, self.bias)

    def extra_repr(self) -> str:
        """
        Returns a string representation of the module's extra configuration.

        Returns:
            str: A string representation of the module's extra configuration.
        """
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
