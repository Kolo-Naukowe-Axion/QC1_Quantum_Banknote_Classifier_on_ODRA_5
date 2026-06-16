"""Best-effort extraction of calibration / job metadata from IQM Qiskit results."""


def calibration_set_id(result) -> str | None:
    """Return calibration set id if the provider exposes it (IQM client versions differ)."""
    for path in (
        lambda r: getattr(r, "parameters", None),
        lambda r: getattr(r, "metadata", None),
        lambda r: getattr(r, "_metadata", None),
    ):
        try:
            obj = path(result)
            if obj is None:
                continue
            if isinstance(obj, dict):
                cid = obj.get("calibration_set_id") or obj.get("calibration_set")
                if cid is not None:
                    return str(cid)
            cid = getattr(obj, "calibration_set_id", None)
            if cid is not None:
                return str(cid)
        except Exception:
            continue
    return None
