# Animal-AI Olympics agent, in PyTorch

Ref. [Denys88/rl_animal](https://github.com/Denys88/rl_animal)

## Install

```bash
uv sync
```

## Train

The player logs one CSV row per step into a queue its writer cannot drain at this
throughput, which puts the host out of memory after a few hours. Disable it once:

```bash
uv run patch_env_logging /path/to/animalAI.x86_64
```

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
