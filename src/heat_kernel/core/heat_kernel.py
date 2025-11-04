import torch

class HeatKernelRegularizer:
    def __init__(self, manifold):
        self.manifold = manifold
