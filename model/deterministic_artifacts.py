from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sorted(arrays):
            arr = np.asanyarray(arrays[name])
            member = io.BytesIO()
            np.lib.format.write_array(member, arr, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, member.getvalue())
    return buf.getvalue()


def save_npz_deterministic(path: str | Path, **arrays: np.ndarray) -> bool:
    target = Path(path)
    payload = npz_bytes(arrays)
    if target.exists() and target.read_bytes() == payload:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)
    return True
