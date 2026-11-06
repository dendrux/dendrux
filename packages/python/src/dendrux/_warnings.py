"""Locate the application call that triggered a library warning."""

import inspect


def _developer_stacklevel() -> int:
    frame = inspect.currentframe()
    stacklevel = 0
    while frame is not None and frame.f_globals.get("__name__", "").startswith("dendrux."):
        stacklevel += 1
        frame = frame.f_back
    del frame
    return stacklevel
