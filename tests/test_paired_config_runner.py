from pathlib import Path

from detr3d.scripts.run_paired_config_search import base_command


def test_confirmation_command_uses_full_confirmation_dataset() -> None:
    manifest = {
        "dataset": {
            "dataroot": "/dataset",
            "version": "v1.0-trainval",
            "train_samples": 1024,
            "val_samples": 512,
        },
        "protocol": {
            "epochs": 60,
            "eval_epochs": [20, 40, 60],
            "image_height": 900,
            "image_width": 1600,
            "num_queries": 900,
            "batch_size": 2,
            "num_workers": 2,
            "lr": 0.0002,
            "backbone_lr_mult": 0.1,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_steps": 17,
            "min_lr_ratio": 0.001,
            "grad_clip_norm": 35.0,
        },
    }

    command = base_command(manifest, Path("/output"), seed=2, phase="confirmation")

    assert command[command.index("--max-samples") + 1] == "1024"
    assert command[command.index("--max-val-samples") + 1] == "512"
    assert command[command.index("--num-eval-samples") + 1] == "512"
    assert command[command.index("--batch-size") + 1] == "2"
    assert command[command.index("--warmup-steps") + 1] == "17"
    assert command[command.index("--seed") + 1] == "2"
    assert command[command.index("--val-split") + 1] == "val"
