import torch
from core.heat_kernel import HeatKernelRegularizer

if __name__ == "__main__":
    reg = HeatKernelRegularizer(n_samples=512)
    theta = torch.tensor([1.0, 1.0])
    tau = torch.tensor(1e-3)

    val, diag = reg.forward(theta, tau)
    print("Regularized Curvature Φτ^reg =", val.item())
    print("Diagnostics =", diag)
  
