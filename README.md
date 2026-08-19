# Animal-AI Olympics agent, in PyTorch

Ref. [Denys88/rl_animal](https://github.com/Denys88/rl_animal)

## Install

```bash
git submodule update --init
uv sync
```

Ruff lint (`--fix`) and ruff-format are applied to `*.py` / `*.pyi` files on every `git commit`.

```bash
uv tool install pre-commit
pre-commit install
```

## Train

The player logs one CSV row per step into a queue its writer cannot drain at this
throughput, which puts the host out of memory after a few hours. Disable it once:

```bash
uv run patch_env_logging
```

```bash
uv run train exp_name
```

## Evaluate

Training ends by sweeping every checkpoint it wrote, so this is only needed to re-score a
run. A run directory scores all of its `ckpt/model_<frame>.pt` and writes the
steps-vs-pass-rate curve they trace to `eval/curve.csv`; a single checkpoint scores just
itself. Both skip nothing and overwrite nothing already scored.

```bash
uv run evaluate results/20260812_213000_exp_name
uv run evaluate results/20260812_213000_exp_name/ckpt/model_best.pt
```

## What a run writes

```
results/<run>/
  config.json     every resolved setting (tracked)
  git_info.txt    the commit and diff it ran from (tracked)
  train_log.csv   one row per epoch (tracked)
  eval/curve.csv  pass rate against frames (tracked)
  eval/<checkpoint>/{detail.csv,summary.csv}   900 arenas, one row each (tracked)
  ckpt/model_<frame>.pt   weights only, 38 MB (not tracked)
  ckpt/trainer_last.pt    weights and optimizer, for --restore (not tracked)
```

## Test

```bash
uv run pytest
```

## License

MIT, as the original.
