![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17527422.svg)

# Manifold-Kernel
**Geometric Heat Kernel Framework on Differentiable Manifolds**

[![CI Tests](https://github.com/zeusindomitable-max/Manifold-Kernel/actions/workflows/tests.yml/badge.svg)](https://github.com/zeusindomitable-max/Manifold-Kernel/actions)

---

## 🔬 Overview
**Manifold-Kernel** is a PyTorch-based framework implementing geometric heat kernel
regularization on differentiable manifolds. It bridges analytical heat-equation
theory with numerical variational optimization, providing a foundation for
curvature-regularized learning systems.

Mathematical foundation: [`docs/mathematical_foundation.md`](docs/mathematical_foundation.md)

---

## ⚗️ Example Usage
```python
import torch
from heat_kernel.geometry.manifolds import S2Manifold
from heat_kernel.core.heat_kernel import HeatKernelRegularizer

manifold = S2Manifold(radius=1.0)
kernel = HeatKernelRegularizer(manifold)

theta = torch.tensor([[0.5, 1.0]])
out = kernel.forward(theta, tau=0.1)
print(out)

## 💝 Support

Love this project? Help me keep building:

**ETH:** ` 0x7cc8686f434cf9b2f274f46fcf73ba6394635b48`

**BTC:** `1LUD9c2hYUERgPmtZCcUitDg8rgrNHfoYP`

**SOL:** `7mp34H3DEdBu5SxWtgkoM6QApYVwKyaY4P1Um7fcnMjZ`


Even small amounts help cover coffee ☕ and server costs!

## ⚙️ `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "manifold-kernel"
version = "0.1.0"
authors = [{ name = "Hari Hardiyan", email = "zeusindomitable@gmail.com" }]
description = "Heat kernel regularizer on S² manifold in PyTorch"
readme = "README.md"
license = { text = "Dual License (MIT/GPLv3)" }
requires-python = ">=3.9"
dependencies = ["torch>=2.0", "numpy"]

[project.urls]
Homepage = "https://github.com/zeusindomitable-max/Manifold-Kernel/tree/main"
Issues = "(https://github.com/zeusindomitable-max/Manifold-Kernel/tree/main)"
```

