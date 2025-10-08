import torch

from loss.components import BasicClassificationLoss, ReconstructionLoss, PercentCorrect


def test_basic_classification_loss() -> None:
    cls_loss = BasicClassificationLoss("cls_loss", 0.5, torch.nn.CrossEntropyLoss())
    pred = {"logits": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])}
    target = {"class": torch.tensor([0, 1])}
    loss = cls_loss(pred, target)
    assert torch.isclose(loss, torch.tensor(1.1519428491592407)).item()


def test_reconstruction_loss() -> None:
    rec_loss = ReconstructionLoss("rec_loss", 0.5, torch.nn.MSELoss(), "reconstruction")
    pred = {"reconstruction": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])}
    target = {"reconstruction": torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])}
    loss = rec_loss(pred, target)
    assert torch.isclose(loss, torch.tensor(0.01)).item()


def test_percent_correct() -> None:
    pc = PercentCorrect("percent_correct", 0.5)
    pred = {"logits": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])}
    target = {"class": torch.tensor([2, 1])}
    loss = pc(pred, target)
    assert torch.isclose(loss, torch.tensor(0.5)).item()
