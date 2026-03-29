from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def plot_series(values: Iterable[float], title: str, ylabel: str, out_path: str | Path) -> None:
    vals: List[float] = list(values)
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(vals)), vals)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
