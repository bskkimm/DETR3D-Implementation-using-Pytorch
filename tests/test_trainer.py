from contextlib import nullcontext

import pytest
import torch

import detr3d.engine.trainer as trainer


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, images, img_metas):
        prediction = self.weight.expand_as(images)
        return {"cls_scores": prediction, "bbox_preds": prediction}


class SquaredErrorCriterion:
    def loss_by_feat(self, cls_scores, bbox_preds, gt_boxes, gt_labels):
        del bbox_preds, gt_labels
        return {"loss_cls": 0.5 * (cls_scores - gt_boxes[0]).square().mean()}


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.step_count = 0
        self.zero_grad_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.zero_grad_count += 1
        return super().zero_grad(set_to_none=set_to_none)


class CountingScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


def make_batches(targets):
    return [
        {
            "images": torch.ones(1),
            "img_metas": [],
            "gt_boxes_ego": [torch.tensor([target], dtype=torch.float32)],
            "gt_labels": [torch.zeros(1, dtype=torch.long)],
        }
        for target in targets
    ]


def test_accumulation_uses_actual_final_window_and_reports_counts(monkeypatch):
    model = ScalarModel()
    optimizer = CountingSGD(model.parameters(), lr=1.0)
    scheduler = CountingScheduler()
    clip_calls = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters, max_norm):
        clip_calls.append(max_norm)
        return original_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    monkeypatch.setattr(
        trainer, "_collect_named_params", lambda model: {"weight": model.weight}
    )
    monkeypatch.setattr(trainer, "_prediction_stats", lambda outputs: {})

    metrics = trainer.train_one_epoch(
        model,
        SquaredErrorCriterion(),
        make_batches([1.0, 2.0, 3.0, 4.0]),
        optimizer,
        torch.device("cpu"),
        grad_clip_norm=100.0,
        debug=True,
        step_scheduler=scheduler,
        accumulation_steps=3,
    )

    assert model.weight.item() == pytest.approx(4.0)
    assert metrics["loss"] == pytest.approx(2.25)
    assert metrics["loss_cls"] == pytest.approx(2.25)
    assert metrics["micro_batches"] == 4.0
    assert metrics["optimizer_steps"] == 2.0
    assert metrics["accumulation_steps"] == 3.0
    assert metrics["debug_grad_weight"] == pytest.approx(2.0)
    assert metrics["debug_delta_weight"] == pytest.approx(2.0)
    assert optimizer.step_count == 2
    assert optimizer.zero_grad_count == 3
    assert scheduler.step_count == 2
    assert clip_calls == [100.0, 100.0]


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_accumulation_steps_must_be_a_positive_integer(value):
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    with pytest.raises(ValueError, match="positive integer"):
        trainer.train_one_epoch(
            model,
            SquaredErrorCriterion(),
            [],
            optimizer,
            torch.device("cpu"),
            accumulation_steps=value,
        )

    with pytest.raises(ValueError, match="positive integer"):
        trainer.fit(
            model,
            SquaredErrorCriterion(),
            [],
            optimizer,
            torch.device("cpu"),
            epochs=0,
            accumulation_steps=value,
        )


def test_fit_forwards_accumulation_steps(monkeypatch):
    captured = []

    def fake_train_one_epoch(**kwargs):
        captured.append(kwargs["accumulation_steps"])
        return {}

    monkeypatch.setattr(trainer, "train_one_epoch", fake_train_one_epoch)
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    trainer.fit(
        model,
        SquaredErrorCriterion(),
        [],
        optimizer,
        torch.device("cpu"),
        epochs=2,
        log_every_epoch=False,
        accumulation_steps=4,
    )

    assert captured == [4, 4]


class OverflowOnceScaler:
    def __init__(self):
        self.scale_value = 8.0
        self.overflow = True

    def scale(self, loss):
        return loss

    def get_scale(self):
        return self.scale_value

    def step(self, optimizer):
        if not self.overflow:
            optimizer.step()

    def update(self):
        if self.overflow:
            self.scale_value /= 2
            self.overflow = False


def test_amp_overflow_skips_optimizer_count_and_scheduler(monkeypatch):
    model = ScalarModel()
    optimizer = CountingSGD(model.parameters(), lr=1.0)
    scheduler = CountingScheduler()
    scaler = OverflowOnceScaler()
    device = type("CudaDevice", (), {"type": "cuda"})()
    monkeypatch.setattr(trainer, "move_batch_to_device", lambda batch, device: batch)
    monkeypatch.setattr(trainer.torch, "autocast", lambda **kwargs: nullcontext())

    metrics = trainer.train_one_epoch(
        model,
        SquaredErrorCriterion(),
        make_batches([1.0, 2.0]),
        optimizer,
        device,
        use_amp=True,
        scaler=scaler,
        step_scheduler=scheduler,
    )

    assert optimizer.step_count == 1
    assert scheduler.step_count == 1
    assert metrics["optimizer_steps"] == 1.0
