from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def elapsed_timer() -> Iterator[callable]:
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
