import torch
import math
from abc import ABC, abstractmethod


class Manifold(ABC):
    @abstractmethod
    def metric(self, theta: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def curvature(self, theta: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def distance(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def volume_element(self, theta: torch.Tensor) -> torch.Tensor:
        pass


class S2Manifold(Manifold):
    def metric(self, theta: torch.Tensor) -> torch.Tensor:
        g11 = torch.ones_like(theta)
        g22 = torch.sin(theta) ** 2
        return torch.stack([g11, g22], dim=-1)

    def curvature(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.full_like(theta, 2.0)

    def distance(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        delta_phi = torch.abs(theta1[..., 1] - theta2[..., 1])
        cos_d = (
            torch.sin(theta1[..., 0]) * torch.sin(theta2[..., 0])
            + torch.cos(theta1[..., 0]) * torch.cos(theta2[..., 0]) * torch.cos(delta_phi)
        )
        cos_d = torch.clamp(cos_d, -1.0, 1.0)
        return torch.arccos(cos_d)

    def volume_element(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.sin(theta[..., 0])
