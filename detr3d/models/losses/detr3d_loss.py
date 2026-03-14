"""Loss adapter."""


class Detr3DLossAdapter:
    """Wraps the current scratch loss behind the new package layout."""

    def __init__(self, loss_impl):
        self.loss_impl = loss_impl

    def loss_by_feat(self, *args, **kwargs):
        return self.loss_impl.loss_by_feat(*args, **kwargs)
