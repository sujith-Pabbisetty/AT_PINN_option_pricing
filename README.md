# PINN Option Pricing:  AT-PINN Reproduction and Extensions

This repository contains a research implementation of an anchored-ensemble
Physics-Informed Neural Network (AT-PINN) for European and American option
pricing, together with two experimental extensions:

1. Activation comparison: `tanh`, GELU, SiLU, and sine.
2. Adaptive training comparison: AT-PINN, SA-PINN, and RAR-PINN.

The experiments use synthetic option-pricing benchmarks. The Yahoo Finance
notebook downloads and cleans market data, but the saved activation/adaptive
comparisons are not real-market validation results.

## Repository contents

| Path | Purpose |
|---|---|
| `01_download_yahoo_full.ipynb` | Download and prepare Yahoo Finance option data |
| `02_at_pinn_full_paper.ipynb` | Main paper-reproduction notebook |
| `02_at_pinn_full_paper.py` | Runnable Python export of the main implementation |
| `03_activation_comparison.py` | Activation-function experiment |
| `04_adaptive_pinn.py` | Adaptive PINN experiments |
| `save_models.py` | Helper for saving trained baseline model weights |
| `data/meta.json` | Market-data snapshot metadata |
| `results/*.csv` | Recorded metrics and comparison tables |
| `results/*.png` | Recorded plots |
| `ppt_content.md` | Slide-by-slide presentation content |

Downloaded CSV data, trained model checkpoints, Python caches, and the paper PDF
are intentionally excluded from new commits. Run the download notebook to
recreate market data as needed. The paper PDF should be obtained from its
publisher or another authorized source; cite the paper in your presentation or
report after confirming its bibliographic details.

## Environment setup

Use Python with a PyTorch build compatible with your machine (CPU or CUDA):

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, install the PyTorch build matching your CUDA setup using the official
PyTorch installation instructions.

## Running the experiments

Run commands from the repository root so relative paths such as `results/`
resolve correctly.

### Paper implementation

Open and run `02_at_pinn_full_paper.ipynb` from top to bottom in Jupyter, or run
the exported script from the repository root:

```bash
python 02_at_pinn_full_paper.py
```

Both default to `MODE = "quick"`; inspect the configuration before choosing
`medium` or `paper`, which require substantially more computation.

### Activation comparison

```bash
python 03_activation_comparison.py
```

The script defaults to `ACT_MODE = "quick"`. It writes comparison metrics,
plots, and model checkpoints under `results/`.

### Adaptive PINN comparison

```bash
python 04_adaptive_pinn.py
```

The script defaults to `ADAPT_MODE = "quick"`. It writes comparison metrics,
plots, and model checkpoints under `results/`.

The scripts define their training budgets and model settings near the top.
Change the mode there before running longer experiments. Outputs may be
overwritten by subsequent runs.

## Recorded results

The checked-in result tables and plots are snapshots from the experiments
already run; rerunning a script may replace them.

### Paper-style benchmark tables

`results/table_1_european_call.csv` and
`results/table_2_american_put.csv` compare the implementation with baselines
and paper-reported values. In the recorded tables, the local PINN-anchor MAE is
approximately `0.1747` for the European-call case and `0.1548` for the
American-put case. The corresponding paper-PINN MAEs recorded in those tables
are `0.067` and `0.045`.

### Extensions

| Experiment | European put MAE | American put MAE |
|---|---:|---:|
| Activation — tanh | 0.2580 | 0.2745 |
| Activation — GELU | 0.1316 | 0.1981 |
| Activation — SiLU | 0.2951 | 0.1974 |
| Activation — sine | 4.9063 | 7.8486 |
| Adaptive — AT-PINN | 0.2580 | 0.2745 |
| Adaptive — SA-PINN | 0.3265 | 0.3147 |
| Adaptive — RAR-PINN | 0.3360 | 0.2099 |

These are single-run, quick-mode results, not averages across random seeds.
GELU had the lowest recorded European-put activation MAE; GELU and SiLU were
close for American puts. RAR-PINN had the lowest recorded American-put MAE,
while AT-PINN performed best among the adaptive methods for European puts.
Treat these as preliminary, setup-specific observations rather than evidence
of universal superiority.

The paper-style benchmark tables and the extension tables use different
experiments and evaluation setups. Their MAEs should not be compared directly.

## Notes before public release

- This project does not include a test suite or claim multi-seed statistical
  significance.
- The paper PDF and Yahoo Finance downloads may be subject to third-party
  copyright or data-use terms; they are not included in new commits.
- `paper.pdf` and the downloaded CSVs are already present in this repository's
  Git history. Adding ignore rules does not erase historical copies. Before
  making this repository public, create a clean repository/history or
  appropriately sanitize the history; do not force-push or rewrite shared
  history without authorization.
- No license is provided. Add one only after deciding what rights you can grant
  for the code and other original materials.
