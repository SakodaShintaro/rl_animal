# Animal-AI Olympics agent, in PyTorch

Ref. [Denys88/rl_animal](https://github.com/Denys88/rl_animal)

## Install

```bash
uv sync
```

## Train

```bash
uv run train \
    --env-path /path/to/animalAI.x86_64 \
    --arenas configs/learning/stage3 \
    --run-name v4_torch
```

The default configuration is the winning run's: 24 actors, 256 steps each, 11400 epochs, or
70 million steps. Checkpoints go to `nn/<run-name>.pt` every epoch and
`nn/<run-name>_best.pt` whenever the mean episode reward improves. Metrics go to wandb;
`--wandb-mode offline` keeps them local, `disabled` drops them.

## Evaluate

```bash
uv run evaluate \
    --env-path /path/to/animalAI.x86_64 \
    --configs configs/learning/competition_configurations \
    --checkpoint nn/v4_torch_best.pt \
    --output results.csv
```

## Test

```bash
uv run pytest
AAI4_ENV_PATH=/path/to/animalAI.x86_64 uv run pytest   # also the ones needing Unity
```

## License

MIT, as the original.
