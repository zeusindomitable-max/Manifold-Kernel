import torch
from heat_kernel.geometry.manifolds import S2Manifold
from heat_kernel.core.heat_kernel import HeatKernelRegularizer

def test_forward_pass():
    manifold = S2Manifold(radius=1.0)
    kernel = HeatKernelRegularizer(manifold)
    theta = torch.tensor([[0.5, 1.0]])
    result = kernel.forward(theta, tau=0.1)
    assert torch.is_tensor(result)
    assert result.shape == (1,)
