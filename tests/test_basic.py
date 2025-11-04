import torch
from core.heat_kernel import HeatKernelRegularizer
from geometry.manifolds import S2Manifold

def test_regulator_converges():
    torch.manual_seed(0)
    reg = HeatKernelRegularizer(n_samples=128)
    theta = torch.tensor([math.pi / 4, math.pi / 4])
    for tau in [1e-2, 1e-3, 5e-4]:
        val, diag = reg.forward(theta, torch.tensor(tau))
        assert abs(val.item() - 2.0) / 2.0 < 0.02, f"Convergence error >2% for τ={tau}"

def test_gradient_stability():
    torch.manual_seed(0)
    reg = HeatKernelRegularizer()
    theta = torch.tensor([math.pi / 3, math.pi / 6], requires_grad=True)
    tau = torch.tensor(1e-3)
    val, _ = reg.forward(theta, tau)
    val.backward()
    grad_norm = torch.norm(theta.grad)
    assert grad_norm < 10.0, "Unstable gradient"
