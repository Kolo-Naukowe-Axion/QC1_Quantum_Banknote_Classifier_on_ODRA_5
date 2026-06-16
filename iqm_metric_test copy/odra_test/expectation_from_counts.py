"""Match `IQMBackendEstimator._counts_to_expectation` in `ansatz_odra_evaluation.ipynb`."""


def counts_to_expectation(counts) -> float:
    if isinstance(counts, list):
        counts = counts[0]
    shots = sum(counts.values())
    count_0 = sum(c for bitstring, c in counts.items() if bitstring[-1] == "0")
    p0 = count_0 / shots if shots else 0.0
    return p0 - (1 - p0)
