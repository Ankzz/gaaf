#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage: ./run_experiments.sh [OPTION...]

Run one or more experiment stages (in dataset, benchmark, baselines, figures order):
  --dataset       Generate the synthetic benchmark dataset
  --benchmark     Run the GAAF benchmark simulation
  --baselines     Run the baseline comparisons
  --figures       Generate the publication figures
  --all           Run every stage above, in order
  -h, --help      Show this help

Run ./setup.sh first to install Python and the project dependencies.
EOF
}

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' "Error: uv was not found on PATH. Install uv and run ./setup.sh first." >&2
    exit 1
fi

run_dataset=false
run_benchmark=false
run_baselines=false
run_figures=false
selected=false

for option in "$@"; do
    case "$option" in
        --dataset)
            run_dataset=true
            selected=true
            ;;
        --benchmark)
            run_benchmark=true
            selected=true
            ;;
        --baselines)
            run_baselines=true
            selected=true
            ;;
        --figures)
            run_figures=true
            selected=true
            ;;
        --all)
            run_dataset=true
            run_benchmark=true
            run_baselines=true
            run_figures=true
            selected=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Error: unknown option: %s\n\n' "$option" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ "$selected" = false ]; then
    usage >&2
    exit 2
fi

if [ "$run_dataset" = true ]; then
    printf '%s\n' "Generating the synthetic benchmark dataset..."
    uv run python -m gaaf.ingestion.pipeline
fi

if [ "$run_benchmark" = true ]; then
    printf '%s\n' "Running the GAAF benchmark..."
    uv run python -m gaaf.experiments.run_benchmark
fi

if [ "$run_baselines" = true ]; then
    printf '%s\n' "Running the baseline comparisons..."
    uv run python -m gaaf.experiments.run_baseline
fi

if [ "$run_figures" = true ]; then
    printf '%s\n' "Generating the publication figures..."
    uv run python -m gaaf.experiments.plot_results
fi
