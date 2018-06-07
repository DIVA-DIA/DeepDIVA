import numpy as np
import torch
from util.evaluation.metrics import accuracy


def test_no_batch():
    # Sanity check
    output = torch.FloatTensor([0.0, 0.0]).unsqueeze(0)
    target = torch.LongTensor([0])
    assert accuracy(output, target)[0].cpu().numpy() == 0.0

    output = torch.FloatTensor([0.0, 1.0]).unsqueeze(0)
    target = torch.LongTensor([1])
    assert accuracy(output, target)[0].cpu().numpy() == 100.0

    output = torch.FloatTensor([0.2, 0.5, 0.7]).unsqueeze(0)
    target = torch.LongTensor([2])
    assert accuracy(output, target)[0].cpu().numpy() == 100.0


def test_mini_batch():
    # Small input
    output = torch.FloatTensor([[0.1, 0.0],
                               [0.1, 0.0]])
    target = torch.LongTensor([0, 0])
    assert accuracy(output, target)[0].cpu().numpy() == 100.0

    # A bit larger input
    output = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                [0.1, 0.7, 0.3, 0.4, 0.5, 0.6],
                                [0.1, 0.2, 0.8, 0.4, 0.5, 0.6],
                                [0.1, 0.2, 0.3, 0.9, 0.5, 0.6],
                                [0.1, 0.2, 0.3, 0.4, 1.0, 0.6],
                                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    target = torch.LongTensor([5, 1, 2, 3, 4, 5])
    assert accuracy(output, target)[0].cpu().numpy() == 100.0

    # A bit larger input - with not 100%
    output = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                [0.1, 0.7, 0.3, 0.4, 0.5, 0.6],
                                [0.1, 0.2, 0.8, 0.4, 0.5, 0.6],
                                [0.1, 0.2, 0.3, 0.9, 0.5, 0.6],
                                [0.1, 0.2, 0.3, 0.4, 1.0, 0.6],
                                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    target = torch.LongTensor([1, 1, 1, 1, 1, 1])
    np.testing.assert_almost_equal(accuracy(output, target)[0].cpu().numpy(), 100/6.0)
