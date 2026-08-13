"""Resource-exhaustion classification shared by formal V6 workers."""

from __future__ import annotations


def is_resource_exhaustion(error: BaseException) -> bool:
    """Return true only for failures that must abort a formal worker."""
    message = str(error).casefold()
    return any(marker in message for marker in (
        "cuda out of memory",
        "out of memory",
        "cublas_status_alloc_failed",
        "resource exhausted",
        "cannot allocate memory",
        "killed",
    ))
