import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl_animal_torch import arena
from rl_animal_torch.config import ENV_PATH
from rl_animal_torch.env import AnimalEnv
from rl_animal_torch.network import AnimalAgent

CONFIGS = "external/animal-ai/configs/competition"
NUM_ENVS = 12
# what PPOTrainer.save_model names its frame-numbered checkpoints
CHECKPOINT_PATTERN = re.compile(r"^model_(\d{9})$")
# far enough above the ports the training instances take that both can be up at once
BASE_PORT = 5900
SEED = 32


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        help="a model_<frame>.pt checkpoint, or a run directory to sweep every checkpoint of",
    )
    parser.add_argument("--configs", default=CONFIGS, help="directory of scenario yaml files")
    parser.add_argument("--num_envs", default=NUM_ENVS, type=int)
    parser.add_argument("--base_port", default=BASE_PORT, type=int)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument(
        "--stride", default=1, type=int, help="take every Nth scenario, for a quick look"
    )
    return parser.parse_args()


def load_agent(path, device):
    state = torch.load(path, map_location=device)
    assert "model" in state, f"{path} carries no weights under 'model'"
    agent = AnimalAgent()
    agent.load_state_dict(state["model"])
    return agent.to(device).eval()


def category_of(path):
    return Path(path).name.split("-")[0]


def build_tasks(paths, num_envs):
    """
    Scenarios are dealt out up front, so a run is reproducible in which environment sees
    which scenario.
    """
    tasks = [[] for _ in range(num_envs)]
    for index, path in enumerate(paths):
        tasks[index % num_envs].append(path)

    return tasks


def summary_row(category, rows):
    passed = [row["passed"] for row in rows]
    return {
        "category": category,
        "episodes": len(rows),
        "passed": sum(passed),
        "pass_rate": round(float(np.mean(passed)), 3),
        "mean_reward": round(float(np.mean([row["reward"] for row in rows])), 3),
    }


def summarize(rows):
    return [summary_row("total", rows)] + [
        summary_row(category, [row for row in rows if row["category"] == category])
        for category in sorted({row["category"] for row in rows})
    ]


def report(summary):
    for entry in summary:
        print(
            f"{entry['category']:<10} pass rate {entry['pass_rate']:.3f} "
            f"({entry['passed']} / {entry['episodes']})  "
            f"mean reward {entry['mean_reward']:.3f}"
        )


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(envs, agent, tasks, writer, device, log_every):
    num_envs = len(envs)
    state = agent.initial_state(num_envs, device=device)
    cursor = [0] * num_envs
    pass_mark = [0.0] * num_envs
    reward = [0.0] * num_envs
    steps = [0] * num_envs
    active = [False] * num_envs
    dones = np.ones(num_envs, dtype=np.float32)
    observations = [None] * num_envs
    rows = []
    total_steps = 0
    start = time.time()

    def start_task(index):
        path = tasks[index][cursor[index]]
        raw = open(path).read()
        pass_mark[index] = arena.read_pass_mark(raw)
        reward[index] = 0.0
        steps[index] = 0
        active[index] = True
        dones[index] = 1.0
        observations[index] = envs[index].reset_to(path)

    for index in range(num_envs):
        start_task(index)

    while any(active):
        running = [index for index in range(num_envs) if active[index]]
        visual = torch.as_tensor(np.asarray([entry[0] for entry in observations]), device=device)
        vels = torch.as_tensor(
            np.asarray([entry[1] for entry in observations], dtype=np.float32), device=device
        )
        logits, _, state = agent(
            visual, vels, state, torch.as_tensor(dones, device=device), num_envs
        )
        actions = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1).cpu().numpy()
        dones = np.zeros(num_envs, dtype=np.float32)

        for index in running:
            envs[index].send(int(actions[index]))
        for index in running:
            observations[index], step_reward, done = envs[index].receive()
            reward[index] += step_reward
            steps[index] += 1
            total_steps += 1
            # No step cap: the arena's `t` is the decay rate of the agent's health, not a
            # step limit, and the player ends the episode when that health runs out. An
            # episode that outlives `t` is one where the agent kept collecting rewards,
            # which refill it. Training never capped either, here or in vla_streaming_rl.
            if not done:
                continue

            path = tasks[index][cursor[index]]
            row = {
                "scenario": Path(path).name,
                "category": category_of(path),
                "pass_mark": pass_mark[index],
                "reward": reward[index],
                # a scenario is passed when the raw episode reward reaches its pass_mark
                "passed": int(reward[index] >= pass_mark[index]),
                "steps": steps[index],
            }
            rows.append(row)
            writer.writerow(row)
            cursor[index] += 1
            if cursor[index] < len(tasks[index]):
                start_task(index)
            else:
                active[index] = False

            if len(rows) % log_every == 0:
                elapsed = time.time() - start
                rate = np.mean([entry["passed"] for entry in rows])
                print(
                    f"{len(rows)} episodes, {total_steps / elapsed:.0f} steps/s, "
                    f"{elapsed / 60.0:.1f} min elapsed, pass rate so far {rate:.3f}",
                    flush=True,
                )

    return rows


def run(checkpoint, configs, num_envs, base_port, seed, stride, output_dir):
    paths = arena.collect([configs])[::stride]
    print(f"scenarios: {len(paths)}, envs: {num_envs}")

    device = torch.device("cuda")
    agent = load_agent(checkpoint, device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "detail.csv"
    summary_output = output_dir / "summary.csv"
    print(f"loaded {checkpoint}, writing {output}")

    envs = [
        AnimalEnv(
            ENV_PATH,
            paths,
            index,
            base_port,
            seed + index,
            shape_rewards=False,
        )
        for index in range(num_envs)
    ]
    fields = ["scenario", "category", "pass_mark", "reward", "passed", "steps"]
    try:
        # written as the episodes end so a run that dies still leaves its results, then
        # rewritten in scenario order once they are all in
        with open(output, "w", buffering=1, newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fields)
            writer.writeheader()
            rows = evaluate(envs, agent, build_tasks(paths, num_envs), writer, device, 25)
    finally:
        for env in envs:
            env.close()

    rows.sort(key=lambda row: row["scenario"])
    write_csv(output, fields, rows)

    summary = summarize(rows)
    write_csv(
        summary_output, ["category", "episodes", "passed", "pass_rate", "mean_reward"], summary
    )
    print(f"wrote {summary_output}")
    report(summary)
    sys.stdout.flush()
    return summary


def eval_dir_for(checkpoint):
    """`<run>/ckpt/model_000253952.pt` scores into `<run>/eval/model_000253952/`."""
    checkpoint = Path(checkpoint).resolve()
    return checkpoint.parent.parent / "eval" / checkpoint.stem


def write_curve(eval_dir):
    """Collect the frame-numbered evaluations into one steps-vs-pass-rate table.

    This is the file the paper's figures read: one row per checkpoint, the overall pass
    rate and the ten per-category ones. `model_best` is left out of it -- it is not a
    point on a training curve, since which epoch it came from moves with the run.
    """
    eval_dir = Path(eval_dir)
    rows = []
    for directory in sorted(eval_dir.iterdir()):
        match = CHECKPOINT_PATTERN.match(directory.name)
        summary_path = directory / "summary.csv"
        if match is None or not summary_path.exists():
            continue

        by_category = {entry["category"]: entry for entry in csv.DictReader(open(summary_path))}
        total = by_category.pop("total")
        row = {
            "frame": int(match.group(1)),
            "episodes": int(total["episodes"]),
            "passed": int(total["passed"]),
            "pass_rate": float(total["pass_rate"]),
            "mean_reward": float(total["mean_reward"]),
        }
        for category in sorted(by_category):
            row[f"pass_rate_{category}"] = float(by_category[category]["pass_rate"])
        rows.append(row)

    assert len(rows) > 0, f"no evaluated checkpoints under {eval_dir}"
    fields = list(rows[0].keys())
    for row in rows:
        assert list(row.keys()) == fields, f"{row['frame']} has categories the others do not"

    curve_path = eval_dir / "curve.csv"
    write_csv(curve_path, fields, rows)
    print(f"wrote {curve_path}")
    for row in rows:
        print(f"frame {row['frame']:>9}  pass rate {row['pass_rate']:.3f}")


def sweep(result_dir, configs, num_envs, base_port, seed, stride):
    """Score every frame-numbered checkpoint of a run, then write their curve.

    Evaluation is 900 arenas per checkpoint, so it is left until training has finished and
    the machine is free. A checkpoint whose summary is already on disk is skipped, which
    is what makes an interrupted sweep resumable.
    """
    checkpoint_dir = Path(result_dir) / "ckpt"
    assert checkpoint_dir.is_dir(), f"{checkpoint_dir} does not exist"

    checkpoints = [
        path for path in sorted(checkpoint_dir.iterdir()) if CHECKPOINT_PATTERN.match(path.stem)
    ]
    assert len(checkpoints) > 0, f"no model_<frame>.pt under {checkpoint_dir}"
    # scored like the rest, but kept out of the curve: see write_curve
    best = checkpoint_dir / "model_best.pt"
    if best.exists():
        checkpoints.append(best)
    print(f"sweeping {len(checkpoints)} checkpoints under {checkpoint_dir}")

    for checkpoint in checkpoints:
        output_dir = eval_dir_for(checkpoint)
        if (output_dir / "summary.csv").exists():
            print(f"skipping {checkpoint.name}: {output_dir} already scored")
            continue
        run(checkpoint, configs, num_envs, base_port, seed, stride, output_dir)

    write_curve(Path(result_dir) / "eval")


def main():
    args = parse_args()
    target = Path(args.target)
    if target.is_dir():
        sweep(target, args.configs, args.num_envs, args.base_port, args.seed, args.stride)
    else:
        run(
            target,
            args.configs,
            args.num_envs,
            args.base_port,
            args.seed,
            args.stride,
            eval_dir_for(target),
        )


if __name__ == "__main__":
    main()
