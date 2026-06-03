# Manual Lightweight Baselines

This folder contains the first follow-up experiment from `further_experiments.md`:
manual lightweight CNN baselines for comparison with the original CNN baseline
and the NAS-discovered model.

The code is intentionally isolated in this folder and only imports the existing
NAS_RFFI data, training, evaluation, and efficiency utilities. It does not
modify the original project files.

## Models

- `small_cnn_half`: half-width residual CNN.
- `small_cnn_quarter`: quarter-width residual CNN.
- `mobile_cnn`: depthwise-separable lightweight CNN.

## Quick Checks

Run model construction and forward-pass checks:

```powershell
python manual_lightweight_baselines\quick_check.py
```

Run a one-epoch smoke experiment on a tiny packet subset:

```powershell
python manual_lightweight_baselines\run_experiment.py --mode smoke --device cpu --skip-efficiency
```

## Full Experiment

Run all manual lightweight baselines with the same default data and training
configuration used by the existing project:

```powershell
python manual_lightweight_baselines\run_experiment.py --mode train-eval
```

Useful options:

```powershell
python manual_lightweight_baselines\run_experiment.py `
  --models small_cnn_half small_cnn_quarter mobile_cnn `
  --output-dir results\manual_lightweight_baselines `
  --epochs 400 `
  --batch-size 32 `
  --eval-batch-size 128
```

Outputs are saved under `results/manual_lightweight_baselines` by default:

- `<model>/<model>.pth`
- `<model>/train_history.json`
- `<model>/test_metrics.json`
- `<model>/model_summary.json`
- `manual_lightweight_summary.json`

