from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.model import Model


class ModelCallback:
    def on_step_begin(self, model: Model, t) -> bool:
        return True

    def on_step_end(self, model: Model, t) -> bool:
        return True
