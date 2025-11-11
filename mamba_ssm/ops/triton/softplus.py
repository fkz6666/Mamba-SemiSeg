import triton
import triton.language as tl
from packaging import version

from triton_config_stub import Config, jit, autotune, heuristics


TRITON3 = 3 >= 3


if TRITON3:
    jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)
else:
    jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt))