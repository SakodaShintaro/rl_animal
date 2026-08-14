import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl_animal_torch import arena
from rl_animal_torch.config import ENV_PATH, EnvConfig
from rl_animal_torch.env import AnimalEnv
from rl_animal_torch.network import AnimalAgent

CONFIGS = "external/animal-ai/configs/competition"
NUM_ENVS = 12
# far enough above the ports the training instances take that both can be up at once
BASE_PORT = 5900
SEED = 32


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--configs", default=CONFIGS, help="directory of scenario yaml files")
    parser.add_argument("--num_envs", default=NUM_ENVS, type=int)
    parser.add_argument("--base_port", default=BASE_PORT, type=int)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument(
        "--stride", default=1, type=int, help="take every Nth scenario, for a quick look"
    )
    return parser.parse_args()


def load_agent(path, device):
    agent = AnimalAgent()
    state = torch.load(path, map_location=device)
    agent.load_state_dict(state["model"] if "model" in state else state)
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
    config = EnvConfig()
    num_envs = len(envs)
    state = agent.initial_state(num_envs, device=device)
    cursor = [0] * num_envs
    pass_mark = [0.0] * num_envs
    max_steps = [0] * num_envs
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
        max_steps[index] = (
            arena.read_arena_time(raw) * config.physics_steps_per_t // config.decision_period + 100
        )
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
            if not done and steps[index] < max_steps[index]:
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


def run(checkpoint, configs, num_envs, base_port, seed, stride):
    paths = arena.collect([configs])[::stride]
    print(f"scenarios: {len(paths)}, envs: {num_envs}")

    device = torch.device("cuda")
    agent = load_agent(checkpoint, device)
    stem = f"{Path(checkpoint).resolve().with_suffix('')}_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    output = Path(f"{stem}.csv")
    summary_output = Path(f"{stem}_summary.csv")
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


def main():
    args = parse_args()
    run(args.checkpoint, args.configs, args.num_envs, args.base_port, args.seed, args.stride)


if __name__ == "__main__":
    main()
