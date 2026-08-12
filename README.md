# Animal-AI Olympics agent, in PyTorch

Ref. [Denys88/rl_animal](https://github.com/Denys88/rl_animal)

## Install

```bash
uv sync
```

## Train

```bash
uv run train --env-path /path/to/animalAI.x86_64
```

The configuration is the winning run's: 24 actors, 256 steps each, 11400 epochs, or 70
million steps. The run is named after the time it started; checkpoints go to `nn/<run>.pt` every epoch and
`nn/<run>_best.pt` whenever the mean episode reward improves. Metrics go to wandb;
`--wandb-mode offline` keeps them local, `disabled` drops them.

## Evaluate

```bash
uv run evaluate \
    --env-path /path/to/animalAI.x86_64 \
    --checkpoint nn/20260812-213000_best.pt \
    --output results.csv
```

## Test

```bash
uv run pytest
AAI4_ENV_PATH=/path/to/animalAI.x86_64 uv run pytest   # also the ones needing Unity
```

## License

MIT, as the original.
