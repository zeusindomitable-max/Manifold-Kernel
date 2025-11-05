import torch

class S2Manifold:
    """
    2-sphere manifold with analytical metric tensor and curvature.
    """

    def __init__(self, radius: float = 1.0):
        self.radius = radius

    def metric(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Metric tensor for S² in spherical coordinates (θ, φ).
        Returns: [batch, 2, 2] tensor with diag(1, sin²θ).
        """
        batch = theta.shape[0]
        g = torch.zeros(batch, 2, 2, device=theta.device)
        g[:, 0, 0] = 1.0
        g[:, 1, 1] = torch.sin(theta[:, 0]) ** 2
        return g * (self.radius ** 2)

    def curvature(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Scalar curvature for S² is R = 2 / r².
        """
        return torch.full((theta.shape[0],), 2.0 / (self.radius ** 2), device=theta.device)
