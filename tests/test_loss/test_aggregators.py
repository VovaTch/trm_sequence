import torch
from loss.aggregators import WeightedSumAggregator


def test_weighted_sum_aggregator(
    loss_aggregator: WeightedSumAggregator,
) -> None:

    # Create some dummy predictions and targets
    pred = {"output1": torch.tensor(0.5), "output2": torch.tensor(0.8)}
    target = {"target1": torch.tensor(1.0), "target2": torch.tensor(0.0)}

    # Calculate the aggregated loss
    loss_output = loss_aggregator(pred, target)

    # Assert the expected values
    assert torch.isclose(loss_output.total, torch.tensor(0.7)).item()
    assert torch.isclose(loss_output.individual["component1"], torch.tensor(1.0)).item()
    assert torch.isclose(loss_output.individual["component2"], torch.tensor(1.0)).item()
    assert torch.isclose(loss_output.individual["component3"], torch.tensor(1.0)).item()
