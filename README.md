# Manifold Kernel
Heat Kernel Regularizer on the 2-Sphere (S²) implemented in PyTorch.

## Features
- Abstract `Manifold` base class
- Concrete implementation for S²
- Heat kernel regularization with Gaussian kernel
- Tests ensuring convergence and gradient stability

## Usage
```bash
pytest -v
python examples/s2_demo.py
```
License

Dual licensed under MIT and GPLv3 — choose whichever suits your use case.
© 2025 Hari Hardiyan

---
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
