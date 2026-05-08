from __future__ import annotations


class EarlyStopping:
    """Tracks validation loss and signals when to stop training.

    The step() method now returns a tuple (should_stop, is_best) so the
    training loop has a single source of truth for both decisions.
    Previously, best-model saving was duplicated outside this class, which
    could diverge on floating-point edge cases.
    """

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best: float | None = None
        self.bad_epochs: int = 0

    def step(self, value: float) -> tuple[bool, bool]:
        """Advance one epoch.

        Returns:
            (should_stop, is_best)
            - should_stop: True when patience is exhausted.
            - is_best:     True when this epoch achieved a new best value.
        """
        is_best = self.best is None or value < self.best
        if is_best:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience
        return should_stop, is_best
