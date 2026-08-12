# Animal-AI Olympics agent, in PyTorch

The agent that won the NeurIPS 2019 Animal AI Olympics, rebuilt in PyTorch and running
against the current Animal-AI environment (v4, Unity 6) instead of the 2019 one.

The original is [Denys88/rl_animal](https://github.com/Denys88/rl_animal): PPO with a
recurrent policy over a residual convolutional tower, TensorFlow 1.14, and the Animal-AI v1
Unity build. `Animal AI Presentation.pdf` is the author's own account of it, including the
learning configuration and the 70 million steps the winning run took.

Nothing here needs TensorFlow, Docker, or the v1 environment; all three have been removed.

## What the move to v4 cost, and what recovering it took

Scored on the 900 released competition scenarios, one episode each:

| | pass rate |
| --- | --- |
| v1 build, the released v1 checkpoint | 44.78% |
| **v4 build**, the released v1 checkpoint | **17.44%** |
| v4 build, retrained on v4 for 3.07M steps (4.4% of the original run) | 38.67% |

The checkpoint that scored 44.78% on the environment it was trained against scores 17.44%
on the new one. `decisionPeriod`, `timescale` and the episode-length bookkeeping each
account for about a point; the rest is that v4 looks different. The same arena, as the
network sees it, in both builds:

![v1 and v4 rendering the same arena](local/aai_v1_vs_v4_1-1-1.png)

The sky, the walls and the floor are different colors, the goal is desaturated, and it
subtends about half the angle it used to at the same distance, so a network that read
distance off apparent size has lost its scale. Retraining on v4 recovers it: 4.4% of the
way through a run, the categories that had collapsed to zero (obstacles, avoidance) are
back, and object permanence and internal models are above what the original managed.

## Install

```bash
uv sync
```

Python is pinned to 3.12. `animalai` declares `Requires-Python >=3.10.12,<3.10.13`, which
uv does not enforce for dependencies and which its pure-python code does not need. What does
need forcing is `protobuf`: `animalai` pins `==3.20.3` while `mlagents-envs` pins `<3.20`,
so `[tool.uv] override-dependencies` settles it on 3.20.3, the version the v4 player was
built against.

The environment binary is not part of this repository. Download the Animal-AI v4 release and
point `--env-path` at its `animalAI.x86_64`.

## Train

```bash
uv run train \
    --env-path /path/to/animalAI.x86_64 \
    --arenas configs/learning/stage3 \
    --run-name v4_torch
```

`stage1`, the default, is the winning run's configuration: 24 actors, 256 steps per actor,
11400 epochs, which is the 70 million steps the presentation reports. `--config stage2` is
what it switched to after the first 50 million (entropy 0.001, learning rate 5e-5, clipping
0.1); reach it by restoring stage1's checkpoint.

```bash
uv run train ... --config stage2 --restore nn/v4_torch.pt --run-name v4_torch_stage2
```

Checkpoints are written to `nn/<run-name>.pt` every epoch and `nn/<run-name>_best.pt`
whenever the mean episode reward improves. Metrics go to wandb; `--wandb-mode offline` keeps
them local and `disabled` drops them. Each epoch also prints a line with the throughput,
elapsed time and an estimate of the time remaining.

Every arena file is checked before a single Unity instance is launched, because a level the
v4 player cannot build does not report an error: it stops answering while writing gigabytes
of exceptions to its log. See `src/rl_animal_torch/arena.py` for the two ways of writing an
arena that do this and why v1 tolerated both.

## Evaluate

```bash
uv run evaluate \
    --env-path /path/to/animalAI.x86_64 \
    --configs /path/to/animal-ai/configs/competition \
    --checkpoint nn/v4_torch_best.pt \
    --output results.csv
```

`--stride 9` takes every ninth scenario, which is 100 of the 900 with ten from each
category: enough to watch a run in progress. Scoring uses the raw environment reward against
each scenario's `pass_mark`, so the numbers are comparable to the competition score.

## Test

```bash
uv run pytest
AAI4_ENV_PATH=/path/to/animalAI.x86_64 uv run pytest   # also the ones needing Unity
```

`tests/test_arena.py` and `tests/test_training.py` need neither Unity nor a GPU and run in
seconds; the training one drives the whole rollout and update against a stand-in
environment. `tests/test_animalai.py` runs a two-epoch training and a three-scenario
evaluation against the real player, and skips itself unless `AAI4_ENV_PATH` is set.

## Layout

| | |
| --- | --- |
| `src/rl_animal_torch/network.py` | the policy: residual tower with channel attention and fixup scalars, a layer-normalized LSTM, value and logits heads |
| `src/rl_animal_torch/ppo.py` | PPO as the original implemented it: GAE, both losses clipped, sequence-wise minibatches, gradient clipping |
| `src/rl_animal_torch/env.py` | one Unity instance, presenting the stacked frames and velocities the network expects |
| `src/rl_animal_torch/vec_env.py` | those instances in worker processes, stepped in lockstep |
| `src/rl_animal_torch/arena.py` | reading, checking and rewriting arena YAML |
| `src/rl_animal_torch/config.py` | the winning run's hyperparameters and the environment settings |
| `configs/learning/` | training levels; `stage3` is the set the first submission used |
| `configs/learning/competition_configurations/` | the 900 competition scenarios |

## Notes on the port

These are the things that had to be right for the PyTorch network to agree with the
TensorFlow one, which it did to a relative 1e-06 before the TensorFlow implementation was
removed. They are worth knowing if the network is ever changed.

- TensorFlow flattens NHWC, so the feature map is permuted back to NHWC before being
  flattened. Otherwise the 4608-wide dense layer reads its input in the wrong order, with
  the shapes still agreeing.
- `'SAME'` padding is asymmetric when the total padding is odd, which it is for the first two
  max pools (84 to 42 and 42 to 21 each need one row and column, placed at the bottom and
  right). `nn.MaxPool2d`'s symmetric padding gives the same output size and shifts every
  pixel.
- The layer-normalized LSTM normalizes the input and recurrent contributions separately over
  the whole `4 * units` axis rather than per gate, adds a third bias afterwards, and
  normalizes the cell state again inside the output gate. Its gate order is i, f, o, u.
- A batch is environment-major: environment `e` at step `t` sits at `e * steps_num + t`.
- Actions were drawn with a Gumbel-max over a TensorFlow random tensor, so only the
  distribution carries over; this samples with `torch.multinomial`.

## License

MIT, as the original.
