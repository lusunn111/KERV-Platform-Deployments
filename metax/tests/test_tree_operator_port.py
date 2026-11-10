"""Check the Moore archive's portable tree operator against our Python planner."""

import pytest
import torch

from openvla.experiments.robot.flagos_tree_ops import (
    build_verification_subtree,
    pack_verification_tokens,
    pack_verification_tokens_padded,
)
from openvla.prismatic.extern.hf.utils import _build_static_verification_plan


@pytest.mark.parametrize("count", [1, 7, 49, 300])
@pytest.mark.parametrize("limit", [0, 1, 48])
def test_portable_tree_matches_python(count, limit):
    generator = torch.Generator().manual_seed(19)
    parents = [0] + [int(torch.randint(i, (), generator=generator)) for i in range(1, count)]
    children = {}
    for node in range(1, count):
        children.setdefault(parents[node], set()).add(node)
    expected = _build_static_verification_plan(
        dict(enumerate(parents)), children, count, limit, torch.float32
    )
    actual = build_verification_subtree(torch.tensor(parents), limit)
    for value, reference in zip(actual[:4], expected[:4]):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert actual[4].tolist() == [
        expected[4][key] for key in ("full_nodes", "verified_nodes", "full_paths", "verified_paths")
    ]
    tokens = torch.arange(2 * count).reshape(2, count)
    packed = tokens[:, actual[0]]
    torch.testing.assert_close(pack_verification_tokens(tokens, actual[0]), packed)
    padded = pack_verification_tokens_padded(tokens, actual[0], packed.shape[1] + 3)
    torch.testing.assert_close(padded[:, :packed.shape[1]], packed)
    torch.testing.assert_close(padded[:, packed.shape[1]:], tokens[:, :1].expand(-1, 3))


@pytest.mark.parametrize("parents", [[0, 2, 1], [0, 1], [0, -1]])
def test_invalid_topology_rejected(parents):
    with pytest.raises(RuntimeError):
        build_verification_subtree(torch.tensor(parents), 0)
