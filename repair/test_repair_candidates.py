import torch
import torch.nn as nn

from repair.repair import _fresh_candidate_state
import test as test_cli


def test_fresh_candidate_state_discards_previous_network_modifications() -> None:
    original = nn.Sequential(nn.Linear(1, 1))
    with torch.no_grad():
        original[0].weight.fill_(2.0)
        original[0].bias.fill_(3.0)
    original_regions = [{"subspace": {"lower": 1.0}}]

    first_network, first_regions = _fresh_candidate_state(
        original, original_regions
    )
    with torch.no_grad():
        first_network[0].weight.fill_(99.0)
        first_network[0].bias.fill_(98.0)
    first_regions[0]["subspace"]["lower"] = 97.0

    next_network, next_regions = _fresh_candidate_state(
        original, original_regions
    )

    assert next_network[0].weight.item() == 2.0
    assert next_network[0].bias.item() == 3.0
    assert next_regions == [{"subspace": {"lower": 1.0}}]


def test_sequential_layer_candidates_are_enabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["test.py"])

    assert test_cli.parse_args().sequential_layer_candidates is True


def test_explicit_layer_selection_uses_targeted_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["test.py", "--repair-layer-config", "m9_16"]
    )

    assert test_cli.parse_args().sequential_layer_candidates is False


def test_sequential_default_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["test.py", "--no-sequential-layer-candidates"]
    )

    assert test_cli.parse_args().sequential_layer_candidates is False


def test_last_layer_aliases_are_resolved_for_each_network() -> None:
    expected = {
        "mnist 9x100": [10, 12, 14, 16],
        "mnist_256x4": [3, 5, 7, 9],
        "cifar10_cnn1": [0, 2, 5, 7],
        "gtsrb_cnn": [0, 2, 5, 7],
    }

    for model_name, layers in expected.items():
        resolved_layers, max_loops = test_cli.resolve_automatic_layer_config(
            model_name, "last_4"
        )
        assert resolved_layers == layers
        assert max_loops == 3


def test_last_layer_alias_uses_configured_loop_limit() -> None:
    layers, max_loops = test_cli.resolve_automatic_layer_config(
        "mnist 9x100", "last_2"
    )

    assert layers == [14, 16]
    assert max_loops == 5
