"""Regression checks for seven-DoF action packing at the KF boundary."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from openvla.prismatic.extern.hf.modeling_speculation import SpecVLAforActionPrediction


@pytest.mark.parametrize("accepted", [0, 2, 6])
def test_kf_preserves_verified_root_and_uses_remaining_dofs(accepted):
    # Distinct values per DoF make a shifted history/decoded slice observable.
    verified = [31890, 31891, 31892, 31893, 31894, 31895, 31745]
    prediction = [31810, 31811, 31812, 31813, 31814, 31815, 31745]
    model = SimpleNamespace(vocab_size=32000, bin_centers=np.zeros(255),
        tokenizer=SimpleNamespace(pad_token_id=32000),
        get_action_dim=lambda: 7, kelman_predict_tokens=lambda: prediction)
    prompt = torch.tensor([[1, 29871]])
    packed, count = SpecVLAforActionPrediction._finalize_action_with_kf(
        model, prompt, 1, torch.tensor([verified]), 0, accepted)
    expected = verified[:accepted+1] + prediction[accepted+1:]
    assert packed[0, prompt.shape[1]:].tolist() == expected
    assert len(expected) == count == 7
