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
uv run patch_env_logging
```

```bash
uv run train
```

## Evaluate

```bash
uv run evaluate --checkpoint results/20260812_213000/best.pt --output results.csv
```

## Test

```bash
uv run pytest
```

## License

MIT, as the original.
