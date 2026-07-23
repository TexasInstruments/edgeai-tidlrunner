# Benchmark Script Index

This directory provides benchmark entry scripts and two internal helper scripts.

## Wrapper scripts

Use wrapper scripts for common preset-based runs:
- `run_benchmark_default_pc.sh`
- `run_benchmark_quick_pc.sh`
- `run_benchmark_accuracy_pc.sh`
- `run_benchmark_sanity_pc.sh`
- `run_benchmark_default_evm.sh`
- `run_benchmark_quick_evm.sh`
- `run_benchmark_accuracy_evm.sh`
- `run_benchmark_sanity_evm.sh`

These wrappers pass preset arguments to internal scripts.

## Internal scripts

Internal scripts contain the shared benchmark command flow:
- `_run_benchmark_pc.sh`
- `_run_benchmark_evm.sh`

Use internal scripts directly only when you want to customize arguments in one place.
