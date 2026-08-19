"""The result layout a run leaves behind, without the player or a checkpoint sweep."""

import csv

import pytest
import torch

from rl_animal_torch.evaluate import eval_dir_for, sweep, write_curve

SUMMARY_FIELDS = ["category", "episodes", "passed", "pass_rate", "mean_reward"]


def write_summary(directory, pass_rate):
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "category": "total",
            "episodes": 900,
            "passed": int(900 * pass_rate),
            "pass_rate": pass_rate,
            "mean_reward": 1.0,
        }
    ]
    for index in range(1, 11):
        rows.append(
            {
                "category": f"{index:02d}",
                "episodes": 90,
                "passed": int(90 * pass_rate),
                "pass_rate": pass_rate,
                "mean_reward": 1.0,
            }
        )
    with open(directory / "summary.csv", "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_eval_dir_is_named_after_the_checkpoint(tmp_path):
    checkpoint = tmp_path / "run" / "ckpt" / "model_000253952.pt"
    assert eval_dir_for(checkpoint) == tmp_path / "run" / "eval" / "model_000253952"


def test_write_curve_orders_by_frame_and_keeps_the_categories(tmp_path):
    write_summary(tmp_path / "model_000500000", 0.2)
    write_summary(tmp_path / "model_000100000", 0.1)
    # best is not a point on a training curve: the epoch it came from moves with the run
    write_summary(tmp_path / "model_best", 0.9)

    write_curve(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "curve.csv")))

    assert [int(row["frame"]) for row in rows] == [100_000, 500_000]
    assert [float(row["pass_rate"]) for row in rows] == [0.1, 0.2]
    assert float(rows[1]["pass_rate_07"]) == 0.2
    assert int(rows[0]["episodes"]) == 900


def test_write_curve_refuses_a_directory_with_nothing_scored(tmp_path):
    with pytest.raises(AssertionError):
        write_curve(tmp_path)


def test_sweep_skips_checkpoints_that_are_already_scored(tmp_path):
    """Scored checkpoints are skipped, which is what makes an interrupted sweep resumable
    and what keeps this from needing the player at all."""
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    for frame in (100_000, 200_000):
        torch.save(
            {"model": {}, "epoch": 1, "frame": frame}, checkpoint_dir / f"model_{frame:09d}.pt"
        )
        write_summary(tmp_path / "eval" / f"model_{frame:09d}", 0.3)
    torch.save({"model": {}, "epoch": 1, "frame": 200_000}, checkpoint_dir / "model_best.pt")
    write_summary(tmp_path / "eval" / "model_best", 0.35)

    sweep(tmp_path, "external/animal-ai/configs/competition", 1, 6000, 0, 1)

    rows = list(csv.DictReader(open(tmp_path / "eval" / "curve.csv")))
    assert [int(row["frame"]) for row in rows] == [100_000, 200_000]
