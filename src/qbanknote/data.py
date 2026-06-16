"""Dataset preparation and cross-validation fold I/O."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo

from qbanknote.paths import find_project_root

DEFAULT_FEATURE_RANGE = (-np.pi / 4, np.pi / 4)


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for numpy, PyTorch, and Python's random module."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _engineer_features(X: np.ndarray) -> np.ndarray:
    variance = X[:, 0].reshape(-1, 1)
    skewness = X[:, 1].reshape(-1, 1)
    interaction = (variance * skewness).astype(np.float32)
    return np.hstack((X, interaction)).astype(np.float32)


def prepare_data(
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    map_labels_to_pm1: bool = False,
):
    """
    Fetch and preprocess the Banknote Authentication dataset.

    Returns scaled features in ``[-π/4, π/4]``. Labels are ``{0, 1}`` by default,
    or ``{-1, 1}`` when ``map_labels_to_pm1=True`` (cross-validation convention).
    """
    banknote = fetch_ucirepo(id=267)
    X = banknote.data.features.to_numpy()
    y = banknote.data.targets.to_numpy().ravel()

    assert X.shape[1] == 4, f"Expected 4 features, got {X.shape[1]}"
    assert set(np.unique(y)) == {0, 1}

    X_expanded = _engineer_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_expanded, y, test_size=test_size, random_state=random_state
    )

    scaler = MinMaxScaler(feature_range=DEFAULT_FEATURE_RANGE)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if map_labels_to_pm1:
        y_train = (2 * y_train - 1).astype(np.float32)
        y_test = (2 * y_test - 1).astype(np.float32)

    return X_train_scaled, X_test_scaled, y_train, y_test


def cv_data_dir(root: Path | None = None) -> Path:
    return find_project_root(root) / "cross_validation" / "Data"


def fold_dir(fold: int, root: Path | None = None) -> Path:
    return cv_data_dir(root) / f"fold_{fold}"


def load_fold_csv(
    fold: int,
    split: str = "test",
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = fold_dir(fold, root) / f"{split}_data.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_fold_arrays(
    fold: int,
    split: str = "test",
    *,
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) feature matrix and targets from a CV fold CSV."""
    df = load_fold_csv(fold, split=split, root=root)
    X = df.drop(columns=["target"]).to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.float32)
    return X, y


def load_fold_train(fold: int, *, root: Path | None = None) -> tuple[np.ndarray, int]:
    """Return (X_train, n) for LED computations."""
    df = load_fold_csv(fold, split="train", root=root)
    X = df.drop(columns=["target"]).to_numpy(dtype=np.float64)
    return X, len(df)


def save_test_indices(indices: np.ndarray, path: str | Path) -> None:
    pd.DataFrame({"index": indices}).to_csv(path, index=False)


def load_test_indices(path: str | Path) -> np.ndarray:
    return pd.read_csv(path)["index"].to_numpy()


def generate_folds(n_splits: int = 5, seed: int = 42, output_dir: Path | None = None) -> None:
    """Generate stratified CV fold CSVs (same logic as Datasplit.ipynb)."""
    banknote = fetch_ucirepo(id=267)
    X = banknote.data.features.to_numpy()
    y = banknote.data.targets.to_numpy().ravel()
    X_expanded = _engineer_features(X)
    y_mapped = (2 * y - 1).astype(np.float32)

    base_dir = output_dir or cv_data_dir()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_expanded, y_mapped)):
        fold_path = base_dir / f"fold_{fold + 1}"
        fold_path.mkdir(parents=True, exist_ok=True)

        X_train, X_test = X_expanded[train_idx], X_expanded[test_idx]
        y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]

        scaler = MinMaxScaler(feature_range=DEFAULT_FEATURE_RANGE)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        train_df = pd.DataFrame(X_train_scaled)
        train_df["target"] = y_train
        train_df.to_csv(fold_path / "train_data.csv", index=False)

        test_df = pd.DataFrame(X_test_scaled)
        test_df["target"] = y_test
        test_df.to_csv(fold_path / "test_data.csv", index=False)
