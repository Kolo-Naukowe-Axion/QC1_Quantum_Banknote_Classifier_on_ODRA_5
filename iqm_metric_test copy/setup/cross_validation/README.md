# Cross-validation data layout

Place fold test CSVs and trained checkpoints here before running experiments.

## Test data

```
Data/
  fold_1/test_data.csv
  fold_2/test_data.csv
  ...
  fold_5/test_data.csv
```

Each CSV must have a header row and six columns: five input features plus one binary label (`0`/`1` or `-1`/`1`).

## Weights

```
Models/Weights/
  depth 2/
    Odra/fold_1/Odra_fold_1_depth_2_epoch_30_weights.pth
    Simulator/fold_1/Simulator_fold_1_depth_2_epoch_30_ideal_weights.pth
  depth 4/
    ...
  depth 6/
    ...
```

Filename patterns are defined in `experiment_lib.weight_path()`. For depth 2, simulator checkpoints use the `_ideal` suffix (`simulator_uses_ideal_suffix = true` in `experiment_config.toml`).

## Copy from QC1

If you have the main QC1 repo checked out locally:

```bash
cp -R ../tests/ansatz_odra_evaluation/setup/cross_validation/Data setup/cross_validation/
cp -R ../tests/ansatz_odra_evaluation/setup/cross_validation/Models/Weights setup/cross_validation/Models/
```

Adjust the source path if your QC1 checkout lives elsewhere.
