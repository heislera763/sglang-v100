"""Use the built SM70 probability filters for speculative verification."""

from sgl_kernel.sampling import (
    _top_k_renorm_probs_internal,
    _top_p_renorm_probs_internal,
)
from sgl_kernel.utils import _to_tensor_scalar_tuple


def top_k_renorm_probs(probs, top_k):
    return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))


def top_p_renorm_probs(probs, top_p):
    return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))
