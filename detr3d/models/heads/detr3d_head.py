"""Head adapter."""

import torch.nn as nn


class Detr3DHeadAdapter(nn.Module):
    """Wraps the current head module behind the new package layout."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, hs):
        return self.head(hs)
