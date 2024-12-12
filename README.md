# NumPy random number generator API for JAX

Small wrapper to use JAX random number generators via the NumPy random API.

    >>> from jrng import JRNG
    >>> rng = JRNG(42)
    >>> rng.standard_normal(3)
    Array([-0.5675502 ,  0.28439185, -0.9320608 ], dtype=float32)
    >>> rng.standard_normal(3)
    Array([ 0.67903334, -1.220606  ,  0.94670606], dtype=float32)
