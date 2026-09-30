"""Small FP16 dense projections use the measured SM70 GEMV alternative."""


def apply_unquant(original, self, layer, x, bias=None):
    from .kernels.sm70_dense_gemv import supported, linear_dense

    if supported(x, layer.weight, bias):
        return linear_dense(x, layer.weight)
    return original(self, layer, x, bias)
