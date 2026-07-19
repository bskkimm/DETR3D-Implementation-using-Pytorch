import random

import numpy as np
import torch

from train import (
    _mlflow_param_updates,
    capture_rng_state,
    restore_rng_state,
    seed_worker,
    set_seed,
)


def test_set_seed_repeats_python_numpy_and_torch_values():
    set_seed(17, deterministic=True)
    first = (random.random(), np.random.rand(), torch.rand(1))

    set_seed(17, deterministic=True)
    second = (random.random(), np.random.rand(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_rng_state_restores_dataloader_generator_and_global_rngs():
    set_seed(3, deterministic=False)
    generator = torch.Generator().manual_seed(3)
    state = capture_rng_state(generator)
    expected = (
        random.random(),
        np.random.rand(),
        torch.rand(1),
        torch.rand(1, generator=generator),
    )

    restore_rng_state(state, generator)
    actual = (
        random.random(),
        np.random.rand(),
        torch.rand(1),
        torch.rand(1, generator=generator),
    )

    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
    assert torch.equal(expected[3], actual[3])


def test_seed_worker_uses_torch_worker_seed(monkeypatch):
    monkeypatch.setattr(torch, "initial_seed", lambda: 123)
    seed_worker(0)
    first = (random.random(), np.random.rand())

    seed_worker(1)
    second = (random.random(), np.random.rand())

    assert first == second


def test_mlflow_resume_only_adds_new_params_and_records_changes():
    additions, changes = _mlflow_param_updates(
        {"epochs": "24", "batch_size": "4"},
        {"epochs": "10", "batch_size": "4", "resume": "checkpoint.pt"},
    )

    assert additions == {"resume": "checkpoint.pt"}
    assert changes == {"epochs": {"original": "24", "resume": "10"}}
