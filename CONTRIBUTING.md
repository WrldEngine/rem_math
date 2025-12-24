# Contributing

Thank you

## Rollout
1. `pip install -r dev-requirements.txt` or alternatively `uv sync` - Installing python dependencies
2. `cargo build` - Installing rust dependencies

## Run benchmarks
- `uv run pytest python/benches/compare_benchmark_test.py` (via uv)
- `pytest python/benches/compare_benchmark_test.py`
