#
# Copyright (c) 2025 Changhe Liu
# Licensed under the Apache License, Version 2.0.
#
# For inquiries contact lch01234@gmail.com
#

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import re


def project_root() -> Path:
    return Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainConfig:
    name: str
    args: dict[str, Any]

    def resolved_args(self, base: dict[str, Any]) -> dict[str, Any]:
        overrides = {**base, **self.args}
        overrides.setdefault(
            "model_path",
            str(project_root() / "output" / "eval" / slugify(self.name)),
        )

        overrides.setdefault(
            "source_path",
            str(project_root().parent / "datasets" / slugify(self.name)),
        )
        return overrides


def slugify(value: str) -> str:
    """replace all illgel characters for folder name"""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower() or "run"


def args_to_cli(arg_mapping: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key, value in arg_mapping.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            # flag args, no value
            if value:
                cli_args.append(flag)
            continue

        cli_args.append(flag)

        if isinstance(value, (list, tuple)):
            cli_args.extend(str(v) for v in value)
        else:
            cli_args.append(str(value))

    return cli_args


def eval_one(conf: TrainConfig, base_args: dict[str, Any]) -> None:
    merged_args = conf.resolved_args(base_args)

    train_script = str(project_root() / "train.py")
    command = [sys.executable, train_script, *args_to_cli(merged_args)]

    print(f"\nRunning Config: {conf.name}")
    print("CLI:", " ".join(command))

    subprocess.run(command, check=True)


def eval_all(confs: Iterable[TrainConfig], base_args=None) -> None:
    if base_args is None:
        base_args = {}
    base_args["eval"] = True

    for conf in confs:
        eval_one(conf, base_args)


if __name__ == "__main__":
    confs: list[TrainConfig] = [
        TrainConfig(
            name="bicycle",
            args={"resolution": 4, "max_shapes": 6400000, "outdoor": True},
        ),
        TrainConfig(
            name="garden",
            args={"resolution": 4, "max_shapes": 5200000, "outdoor": True},
        ),
        TrainConfig(
            name="stump",
            args={"resolution": 4, "max_shapes": 4750000, "outdoor": True},
        ),
        TrainConfig(
            name="bonsai",
            args={"resolution": 2, "max_shapes": 3000000},
        ),
        TrainConfig(
            name="counter",
            args={"resolution": 2, "max_shapes": 2500000},
        ),
        TrainConfig(
            name="room",
            args={"resolution": 2, "max_shapes": 2100000},
        ),
        TrainConfig(
            name="kitchen",
            args={"resolution": 2, "max_shapes": 2400000},
        ),
        TrainConfig(
            name="train",
            args={"resolution": 1, "max_shapes": 2500000, "outdoor": True},
        ),
        TrainConfig(
            name="truck",
            args={"resolution": 1, "max_shapes": 2000000, "outdoor": True},
        ),
    ]

    eval_all(confs)
