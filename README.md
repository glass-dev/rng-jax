# NumPy random number generator API for JAX

**This is a proof of concept only.**

Wraps stateless JAX random number generation in the
[`numpy.random.Generator`](generator) API.

```py
from jrng import JRNG

rng = JRNG(42)

rng.standard_normal(3)
# Array([-0.5675502 ,  0.28439185, -0.9320608 ], dtype=float32)
rng.standard_normal(3)
# Array([ 0.67903334, -1.220606  ,  0.94670606], dtype=float32)
```

The goal of this experiment is to investigate ways in which there can be a
random number generation API that works in tandem with the Python Array API.

## How it works

The `JRNG` class works in the obvious way: it keeps track of the JAX `key` and
calls `jax.random.split()` before every random operation.

## JIT

The problem with a stateful RNG is that it cannot easily be passed into a
compiled function. However, the `JRNG` class is only "stateful" in that it
keeps track of the current `key`.

When a `JRNG` pytree is flattened, the resulting child node contains an
_independent_ random key, while the internal state of the existing `JRNG` is
advanced at the same time. This allows passing `JRNG` instances into compiled
functions and still obtaining independent random outputs:

```py
import jax
from jrng import JRNG

def print_key(key): print(jax.random.key_data(key))

@jax.jit
def f(x, rng):
    return x + rng.standard_normal(x.shape)

x = jax.numpy.array([1, 2, 3])
rng = JRNG(42)

print_key(rng.key)  # [ 0 42]

print(f(x, rng))  # [0.047065  1.6797752 3.9650078]

print_key(rng.key)  # [4249898905 2425127087]

print(f(x, rng))  # [0.60631436 1.0040649  2.4605024 ]

print_key(rng.key)  # [ 499334550 3925197703]
```

However, this mechanism means flattening the `JRNG` pytree changes internal
state (due to the internal details of JAX, it actually advances the random
number generator multiple times).

```py
# same initial state as above
key = jax.random.key(42)
print_key(key)  # [ 0 42]

# pytree is flattened 4 times per invocation
key, _ = jax.random.split(key)
key, _ = jax.random.split(key)
key, _ = jax.random.split(key)
key, _ = jax.random.split(key)
print_key(key)  # [4249898905 2425127087]
```

While this is not an ideal solution, it may be an acceptable one: the goal of
this API is to work in tandem with the Array API. Array-agnostic code is not
usually compiled at low level. Using the `JRNG` class _inside_ a compiled
function works without issue.

[generator]: https://numpy.org/doc/stable/reference/random/generator.html
