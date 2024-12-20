import jax
from jax import dtypes, random
from jax import numpy as jnp
import pytest

import rng_jax


def test_init():
    rng = rng_jax.Generator(42)
    assert isinstance(rng, rng_jax.Generator)
    assert isinstance(rng._key, jax.Array)
    assert dtypes.issubdtype(rng._key.dtype, dtypes.prng_key)
    assert jnp.all(rng._key == random.key(42))


def test_from_key():
    key = random.key(42)
    rng = rng_jax.Generator.from_key(key)
    assert rng._key is key

    with pytest.raises(ValueError, match="not a random key"):
        rng_jax.Generator.from_key(object())

    with pytest.raises(ValueError, match="not a random key"):
        rng_jax.Generator.from_key(jnp.zeros(()))


def test_key():
    rng = rng_jax.Generator(42)
    rngkey, outkey = random.split(rng._key, 2)
    key = rng.key()
    assert jnp.all(rng._key == rngkey)
    assert jnp.all(key == outkey)


def test_spawn():
    rng = rng_jax.Generator(42)
    key, *subkeys = random.split(rng._key, 4)
    subrngs = rng.spawn(3)
    assert rng._key == key
    assert isinstance(subrngs, list)
    assert len(subrngs) == 3
    for subrng, subkey in zip(subrngs, subkeys):
        assert isinstance(subrng, rng_jax.Generator)
        assert subrng._key == subkey


def test_integers():
    rng = rng_jax.Generator(42)
    key = rng._key
    rvs = rng.integers(0, 10, 10000)
    assert rng._key != key
    assert rvs.shape == (10000,)
    assert rvs.min() == 0
    assert rvs.max() == 9


def test_random():
    rng = rng_jax.Generator(42)
    key = rng._key
    rvs = rng.random(10000)
    assert rng._key != key
    assert rvs.shape == (10000,)
    assert rvs.min() >= 0.0
    assert rvs.max() < 1.0


def test_choice():
    rng = rng_jax.Generator(42)
    key = rng._key
    a = jnp.array([1, 2, 3])
    rvs = rng.choice(a, 10000)
    assert rng._key != key
    assert rvs.shape == (10000,)
    assert (jnp.unique(rvs) == a).all()


def test_bytes():
    rng = rng_jax.Generator(42)
    key = rng._key
    rvs = rng.bytes(12)
    assert rng._key != key
    assert isinstance(rvs, bytes)
    assert len(rvs) == 12


def test_permutation():
    rng = rng_jax.Generator(42)
    key = rng._key
    rvs = rng.permutation(100)
    assert rng._key != key
    assert (jnp.unique(rvs) == jnp.arange(100)).all()
