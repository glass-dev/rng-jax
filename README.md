# `rng-jax` — NumPy random number generator API for JAX

**This is a proof of concept only.**

Wraps JAX's stateless random number generation in a class implementing the
[`numpy.random.Generator`](generator) interface.

## Example

```py
>>> import rng_jax
>>> rng = rng_jax.Generator(42)  # same arguments as jax.random.key()
>>> rng.standard_normal(3)
Array([-0.5675502 ,  0.28439185, -0.9320608 ], dtype=float32)
>>> rng.standard_normal(3)
Array([ 0.67903334, -1.220606  ,  0.94670606], dtype=float32)
```

## Rationale

The [Array API](array_api) makes it possible to write array-agnostic Python
libraries. The `rng-jax` package makes it easy to extend this to random number
generation in NumPy and JAX. End users only need to provide a `rng` object, as
usual, which can either be a NumPy one or a `rng_jax.Generator` instance
wrapping JAX's stateless random number generation.

## How it works

The `rng_jax.Generator` class works in the obvious way: it keeps track of the
JAX `key` and calls `jax.random.split()` before every random operation.

## JIT and native JAX code

The problem with a stateful RNG is that it cannot be passed into a compiled JAX
function. In practice, this is not usually an issue, since the goal of this
package is to work in tandem with the Array API: array-agnostic code is not
usually compiled at low level. Conversely, native JAX code usually expects a
`key`, anyway, not a `rng_jax.Generator` instance.

To interface with a native JAX function expecting a `key`, use the `.split()`
method to obtain a new random key and advance the internal state of the
generator:

```py
>>> import jax
>>> rng = rng_jax.Generator(42)
>>> key = rng.split()
>>> jax.random.normal(key, 3)
Array([-0.5675502 ,  0.28439185, -0.9320608 ], dtype=float32)
>>> key = rng.split()
>>> jax.random.normal(key, 3)
Array([ 0.67903334, -1.220606  ,  0.94670606], dtype=float32)
```

Using the `rng_jax.Generator` class fully _within_ a compiled JAX function
works without issue.

[array-api]: https://data-apis.org/array-api/latest/
[generator]: https://numpy.org/doc/stable/reference/random/generator.html
