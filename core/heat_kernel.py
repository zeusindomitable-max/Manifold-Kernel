import torch
import math
from typing import Tuple, Dict
from geometry.manifolds import S2Manifold


class HeatKernelRegularizer:
    def __init__(self, n_samples: int = 256, seed: int = 42):
        self.n_samples = n_samples
        self.manifold = S2Manifold()
        torch.manual_seed(seed)

    def forward(self, theta: torch.Tensor, tau: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        assert tau > 0, "tau must be positive"
        samples = torch.rand((self.n_samples, 2), device=theta.device)
        samples[:, 0] = math.pi * samples[:, 0]
        samples[:, 1] = 2 * math.pi * samples[:, 1]

        dists = torch.stack(
            [self.manifold.distance(theta, s) for s in samples], dim=0
        )
        kernel_vals = (4 * math.pi * tau) ** (-1) * torch.exp(-dists**2 / (4 * tau))

        vol = torch.mean(self.manifold.volume_element(samples))
        reg_curv = torch.mean(kernel_vals * self.manifold.curvature(samples)) * vol

        diagnostics = {
            "mean_kernel": kernel_vals.mean().item(),
            "max_kernel": kernel_vals.max().item(),
            "vol_element": vol.item(),
        }
        return reg_curv, diagnostics
