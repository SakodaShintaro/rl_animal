# PyTorch-only path (Animal-AI v4)

This runs the winning agent without TensorFlow and without Docker. The network is a port
of `networks.animal_a2c_network_lstm6`, checked against the TensorFlow 1.15 checkpoint to a
relative difference of 1e-06, which is the precision a float32 reference allows.

The Docker path is still there and still needed for the original TensorFlow checkpoints:
the only TensorFlow build that runs on Blackwell is `1.15.5+nv23.3`, which NVIDIA ships as
a container image and not as a wheel (their public index stops at `nv22.12`).

## Install

```bash
uv sync
```

`requires-python` is `>=3.12,<3.13`. `animalai` declares `Requires-Python >=3.10.12,<3.10.13`,
which uv does not enforce for dependencies, and its code is pure python with no 3.10-only
syntax. What does need forcing is `protobuf`: `animalai` pins `==3.20.3` while
`mlagents-envs` pins `<3.20`, so `[tool.uv] override-dependencies` settles it on 3.20.3,
the version the v4 player was built against. It installs on 3.12 as a pure-python wheel.

## Train

```bash
uv run train \
    --env-path /path/to/animalai_env/Linux/animalAI.x86_64 \
    --arenas configs/learning/stage3 \
    --run-name v4_torch
```

`stage1` is the winning run's configuration: 24 actors, 256 steps, 11400 epochs, which is
the 70 million steps the presentation reports. `--config stage2` is what it switched to
after the first 50 million steps (entropy 0.001, learning rate 5e-5, clipping 0.1); run it
by restoring stage1's checkpoint:

```bash
uv run train ... --config stage2 --restore nn/v4_torch.pt --run-name v4_torch_stage2
```

`--init-weights local/tf_reference.npz` starts from the TensorFlow checkpoint's weights
instead of a fresh network. That npz comes from `export_tf_reference.py`, which has to run
inside the TensorFlow container:

```bash
docker compose run --rm animal-aai4 python export_tf_reference.py \
    --checkpoint nn/last84_10_5 --output local/tf_reference.npz \
    --env-num 2 --steps-num 1 --seed 0
```

Checkpoints go to `nn/<run-name>.pt` every epoch and `nn/<run-name>_best.pt` whenever the
mean episode reward improves. Logs go to `runs/<run-name>` for tensorboard.

## Evaluate

```bash
uv run evaluate \
    --env-path /path/to/animalai_env/Linux/animalAI.x86_64 \
    --configs /path/to/animal-ai/configs/competition \
    --checkpoint nn/v4_torch_best.pt \
    --output results.csv
```

`--stride 9` takes every ninth scenario, which is 100 of the 900 with ten from each
category: enough to watch a run in progress without waiting for a full pass.
`--checkpoint` also accepts the TensorFlow npz directly.

## Verify the port

```bash
uv run python verify_torch_parity.py --reference local/tf_reference.npz \
    --env-num 2 --steps-num 1
```

Run it with `--steps-num` above 1 as well: that is what exercises the recurrent unrolling
and the environment-major batch layout, which is where a port of this network goes wrong
without being obviously broken.

## Things the port has to get right

- TensorFlow flattens NHWC, so the feature map is permuted back to NHWC before being
  flattened. Otherwise the 4608-wide dense layer reads its inputs in the wrong order, with
  the shapes still agreeing.
- `'SAME'` padding is asymmetric when the total padding is odd, which it is for the first
  two max pools (84 to 42 and 42 to 21 each need one row and column, placed at the bottom
  and right). `nn.MaxPool2d`'s symmetric padding gives the same output size and shifts
  every pixel.
- The layer-normalised LSTM normalises the input and recurrent contributions separately
  over the whole `4 * units` axis rather than per gate, adds a third bias afterwards, and
  normalises the cell state again inside the output gate. Its gate order is i, f, o, u.
- A batch is environment-major: environment `e` at step `t` is at `e * steps_num + t`.
- Actions cannot match. The original samples with a Gumbel-max over a TensorFlow random
  tensor, so only the distribution carries over; the port samples with `torch.multinomial`.

## Arena files

`arena.collect` refuses a level before any Unity instance is launched, because two ways of
writing an arena make the v4 player hang rather than report anything:

- A list key with no value (`colors:` followed by another key) deserialises to null and
  overwrites the C# side's default empty list. `initVec3sFromRGBs` iterates it, and
  `ArenasConfigurations.UpdateWithConfigurationsReceived` neither null-checks nor catches,
  so the configuration dictionary stays empty and `TrainingArena.SetNextArenaID` divides by
  a zero arena count on every FixedUpdate.
- An `Agent` item with no `positions`. `ArenaBuilders.InstantiateSpawnables` reads
  `positions[0]` unconditionally. In v1 an item with no positions was placed at random; the
  way to ask v4 for that is `!Vector3 {x: -1, y: 0, z: -1}`.

Both were accepted by v1, which parsed the YAML in python and normalised null to `[]`.
Neither is visible from python: the symptoms are a connection timeout and a player log
growing by gigabytes. `configs/learning/stage3` has been corrected for both.
