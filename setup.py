from setuptools import setup, find_packages

setup(
    name="manifold_kernel",
    version="1.0.0",
    author="Hari Hardiyan",
    author_email="",
    description="Heat-Kernel Regularization Framework on Differentiable Manifolds",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Dual License (Non-Commercial Free, Commercial Requires Royalty)",
    url="https://github.com/zeusindomitable-max/Manifold-Kernel",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0"
    ],
    python_requires=">=3.9",
)
