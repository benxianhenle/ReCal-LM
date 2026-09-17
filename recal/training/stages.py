"""A round is one evaluation on the same held-out data, not a noisy training batch."""
import math
from dataclasses import dataclass, asdict


@dataclass
class PlateauSwitch:
    patience: int = 5
    relative_tolerance: float = 0.001
    previous: float | None = None
    stable_changes: int = 0
    switched: bool = False

    def observe(self, loss):
        if not math.isfinite(loss):
            raise ValueError('Non-finite validation loss')
        if self.switched:
            return False
        if self.previous is not None:
            relative_change = abs(loss - self.previous) / max(abs(self.previous), 1e-8)
            self.stable_changes = self.stable_changes + 1 if relative_change <= self.relative_tolerance else 0
        self.previous = float(loss)
        if self.stable_changes >= self.patience:
            self.switched = True
            return True
        return False

    def state_dict(self):
        return asdict(self)
