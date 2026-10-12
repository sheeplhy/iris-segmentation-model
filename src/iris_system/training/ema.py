from __future__ import annotations

from contextlib import contextmanager

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if not value.is_floating_point():
                self.state[name].copy_(value)
            else:
                self.state[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.state = {name: value.detach().clone() for name, value in state.items()}

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Temporarily evaluate the exact EMA weights that inference deploys."""
        backup = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.state, strict=True)
        try:
            yield model
        finally:
            model.load_state_dict(backup, strict=True)
