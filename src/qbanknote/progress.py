"""Optional tqdm progress bars with a simple print fallback."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

ProgressCallback = Callable[[str, int, int], None]


def _try_tqdm():
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]

        return tqdm
    except ImportError:
        return None


class _PrintProgress:
    def __init__(self, iterable: Iterable[T], *, total: int | None, desc: str, unit: str) -> None:
        self._iterable = iter(iterable)
        self._total = total
        self._desc = desc
        self._unit = unit
        self._index = 0

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        value = next(self._iterable)
        self._index += 1
        if self._total is not None:
            print(f"{self._desc}: {self._index}/{self._total} {self._unit}", flush=True)
        else:
            print(f"{self._desc}: {self._index} {self._unit}", flush=True)
        return value

    def update(self, n: int = 1) -> None:
        self._index += n
        if self._total is not None:
            print(f"{self._desc}: {self._index}/{self._total} {self._unit}", flush=True)
        else:
            print(f"{self._desc}: {self._index} {self._unit}", flush=True)

    def close(self) -> None:
        return None


def progress_bar(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
):
    """Return a tqdm bar when available, otherwise a minimal print-based wrapper."""
    if disable:
        return iterable

    tqdm = _try_tqdm()
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit=unit)

    if iterable is None:
        return _PrintProgress(range(total or 0), total=total, desc=desc, unit=unit)
    return _PrintProgress(iterable, total=total, desc=desc, unit=unit)


def report_progress(
    callback: ProgressCallback | None,
    stage: str,
    completed: int,
    total: int,
) -> None:
    if callback is not None:
        callback(stage, completed, total)


def make_print_callback(prefix: str = "") -> ProgressCallback:
    def _callback(stage: str, completed: int, total: int) -> None:
        label = f"{prefix}{stage}" if prefix else stage
        print(f"{label}: {completed}/{total}", flush=True)

    return _callback
