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

## Evaluate

```bash
uv run evaluate \
    --env-path /path/to/animalAI.x86_64 \
    --checkpoint results/20260812-213000/best.pt \
    --output results.csv
```

## Test

```bash
uv run pytest
AAI4_ENV_PATH=/path/to/animalAI.x86_64 uv run pytest   # also the ones needing Unity
```

## License

MIT, as the original.
