# gaaf
Governance-Aware Agent Arbitration Framework (GAAF) for Resilient, Decentralized, and Market-Driven Microgrids

## Run the experiments from scratch

The experiments use Python 3.14 and [`uv`](https://docs.astral.sh/uv/). From a
fresh checkout, install `uv` if it is not already available, then run the
setup script from the repository root:

```bash
git clone https://github.com/Ankzz/gaaf.git
cd gaaf
./setup.sh
```

The script installs Python 3.14 with `uv`, creates the project environment,
and installs the exact dependencies recorded in `uv.lock`.

Generate the reproducible, seven-day synthetic microgrid dataset, then run
the GAAF benchmark and baseline comparisons:

```bash
uv run python -m gaaf.ingestion.pipeline
uv run python -m gaaf.experiments.run_benchmark
uv run python -m gaaf.experiments.run_baseline
```

The dataset is written to `data/microgrid_benchmark_7day.csv`. The benchmark
and baseline scripts print their results to the terminal. You can also run
these stages with the root-level experiment runner:

```bash
./run_experiments.sh --dataset
./run_experiments.sh --benchmark --baselines
```

Generate the publication figures after the simulation runs:

```bash
./run_experiments.sh --figures
```

This writes `fig1_battery_soc_sla.png`, `fig2_fdia_event_override.png`, and
`fig3_comparative_metrics.png` to `figures/`. The first two figures use
simulation traces from the generated dataset; the comparative metrics in the
third figure are defined in the plotting script. To execute all stages from
scratch in the correct order, run:

```bash
./run_experiments.sh --all
```

Use `./run_experiments.sh --help` to list the available options.
