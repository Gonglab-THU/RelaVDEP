import torch
from . import soft_rank_numpy

def wrap_class(cls, **kwargs):
    """Wraps the given NumpyOp in a torch Function."""

    class NumpyOpWrapper(torch.autograd.Function):
        """A torch Function wrapping a NumpyOp."""

        @staticmethod
        def forward(ctx, values):
            obj = cls(values.detach().numpy(), **kwargs)
            ctx.numpy_obj = obj
            return torch.from_numpy(obj.compute())

        @staticmethod
        def backward(ctx, grad_output):
            return torch.from_numpy(ctx.numpy_obj.vjp(grad_output.numpy()))

    return NumpyOpWrapper


def map_tensor(map_fn, tensor):
    return torch.stack([map_fn(tensor_i) for tensor_i in torch.unbind(tensor)])


def soft_rank(values, direction="ASCENDING", regularization_strength=1.0,
              regularization="l2"):
    r"""Soft rank the given values (tensor) along the second axis.

    The regularization strength determines how close are the returned values
    to the actual ranks.

    Args:
      values: A 2d-tensor holding the numbers to be ranked.
      direction: Either 'ASCENDING' or 'DESCENDING'.
      regularization_strength: The regularization strength to be used. The smaller
      this number, the closer the values to the true ranks.
      regularization: Which regularization method to use. It
        must be set to one of ("l2", "kl", "log_kl").
    Returns:
      A 2d-tensor, soft-ranked along the second axis.
    """
    if len(values.shape) != 2:
        raise ValueError("'values' should be a 2d-tensor "
                         "but got %r." % values.shape)

    wrapped_fn = wrap_class(soft_rank_numpy.SoftRank,
                            regularization_strength=regularization_strength,
                            direction=direction,
                            regularization=regularization)
    return map_tensor(wrapped_fn.apply, values)


def soft_sort(values, direction="ASCENDING",
              regularization_strength=1.0, regularization="l2"):
    r"""Soft sort the given values (tensor) along the second axis.

    The regularization strength determines how close are the returned values
    to the actual sorted values.

    Args:
      values: A 2d-tensor holding the numbers to be sorted.
      direction: Either 'ASCENDING' or 'DESCENDING'.
      regularization_strength: The regularization strength to be used. The smaller
      this number, the closer the values to the true sorted values.
      regularization: Which regularization method to use. It
        must be set to one of ("l2", "log_kl").
    Returns:
      A 2d-tensor, soft-sorted along the second axis.
    """
    if len(values.shape) != 2:
        raise ValueError("'values' should be a 2d-tensor "
                         "but got %s." % str(values.shape))

    wrapped_fn = wrap_class(soft_rank_numpy.SoftSort,
                            regularization_strength=regularization_strength,
                            direction=direction,
                            regularization=regularization)

    return map_tensor(wrapped_fn.apply, values)
