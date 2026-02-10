from __future__ import annotations

from typing import Optional, Union
import warnings

import torch


def resolve_device(device: Optional[Union[str, torch.device]] = "auto") -> torch.device:
    """
    Resolve a user-provided device specifier into a concrete torch.device.

    Falls back to CPU if CUDA is unavailable.
    """

    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        requested = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        requested = torch.device(str(device))

    if requested.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "CUDA requested for TIVelo but not available; falling back to CPU.",
            RuntimeWarning,
        )
        return torch.device("cpu")

    return requested
