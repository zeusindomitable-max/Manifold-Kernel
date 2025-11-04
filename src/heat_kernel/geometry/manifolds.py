import torch
from typing import Tuple

class S2Manifold:
    """
    Representasi manifold 2-sphere (S^2) dengan koordinat bola (θ, φ).
    θ ∈ [0, π], φ ∈ [0, 2π].
    Metric tensor: diag(r^2, r^2 sin^2 θ)
    Scalar curvature: R = 2 / r^2
    """

    def __init__(self, radius: float = 1.0):
        self.radius = radius

    def metric(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Return metric tensor g_ij untuk S^2.
        Args:
            theta (torch.Tensor): shape [batch_size, 2], berisi (θ, φ)
        Returns:
            torch.Tensor: shape [batch_size, 2, 2], metric tensor diag(r^2, r^2 sin^2 θ)
        """
        if theta.ndim != 2 or theta.shape[1] != 2:
            raise ValueError("Input harus berbentuk [batch_size, 2] untuk (θ, φ).")

        th = theta[:, 0]
        g11 = (self.radius ** 2) * torch.ones_like(th)
        g22 = (self.radius ** 2) * torch.sin(th) ** 2

        batch_size = theta.shape[0]
        g = torch.zeros(batch_size, 2, 2, dtype=theta.dtype, device=theta.device)
        g[:, 0, 0] = g11
        g[:, 1, 1] = g22
        return g

    def curvature(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Return scalar curvature R = 2 / r^2 (konstan untuk S^2).
        """
        return (2.0 / (self.radius ** 2)) * torch.ones(theta.shape[0], dtype=theta.dtype, device=theta.device)

    def distance(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Hitung jarak great-circle antara dua titik di S^2.
        Args:
            theta1, theta2: [batch_size, 2] (θ, φ)
        Returns:
            torch.Tensor: [batch_size] jarak di sepanjang permukaan bola
        """
        if theta1.shape != theta2.shape or theta1.shape[1] != 2:
            raise ValueError("Input harus berbentuk [batch_size, 2] dan sama bentuknya.")

        th1, ph1 = theta1[:, 0], theta1[:, 1]
        th2, ph2 = theta2[:, 0], theta2[:, 1]

        # rumus haversine untuk 2-sphere
        delta_phi = ph2 - ph1
        cos_gamma = (
            torch.sin(th1) * torch.sin(th2) * torch.cos(delta_phi)
            + torch.cos(th1) * torch.cos(th2)
        )
        cos_gamma = torch.clamp(cos_gamma, -1.0, 1.0)  # numerically stable
        gamma = torch.acos(cos_gamma)
        return self.radius * gamma

    def volume_element(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Return elemen volume (Jacobian) untuk integrasi di S^2.
        dV = r^2 sin θ dθ dφ
        """
        if theta.ndim != 2 or theta.shape[1] != 2:
            raise ValueError("Input harus berbentuk [batch_size, 2] untuk (θ, φ).")

        th = theta[:, 0]
        return (self.radius ** 2) * torch.sin(th)

    def dimension(self) -> int:
        """Dimensi manifold (2 untuk S^2)."""
        return 2
      
