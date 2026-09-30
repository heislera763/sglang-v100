"""Small FP16 dense projections use the measured SM70 GEMV alternative."""


def apply(original, self, layer, x, bias=None):
    from .kernels.sm70_dense_gemv import supported, linear

    if supported(x, layer.weight, bias):
        return linear(x, layer.weight)
    return original(self, layer, x, bias)
