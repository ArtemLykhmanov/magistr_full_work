# =============================================
# FILE: src/config.py
# =============================================
from dataclasses import dataclass
from pathlib import Path


SEED = 42


@dataclass
class Paths:
    root: Path
    data: Path
    out: Path


    @classmethod
    def from_root(cls, root: str = "."):
        p = Path(root).resolve()
        return cls(root=p, data=p / "data", out=p / "out")


