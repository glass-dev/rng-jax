import jax
from jax import dtypes, random
from jax import numpy as jnp
import pytest

from jrng import JRNG


def test_init():
    rng = JRNG(42)
    assert isinstance(rng, JRNG)
    assert isinstance(rng.key, jax.Array)
    assert dtypes.issubdtype(rng.key.dtype, dtypes.prng_key)
    assert rng.key == random.key(42)


def test_from_key():
    key = random.key(42)
    rng = JRNG.from_key(key)
    assert rng.key is key

    with pytest.raises(ValueError, match="not a random key"):
        JRNG.from_key(object())

    with pytest.raises(ValueError, match="not a random key"):
        JRNG.from_key(jnp.zeros(()))


def test_spawn():
    rng = JRNG(42)
    key, *subkeys = random.split(rng.key, 4)
    subrngs = rng.spawn(3)
    assert rng.key == key
    assert isinstance(subrngs, list)
    assert len(subrngs) == 3
    for subrng, subkey in zip(subrngs, subkeys):
        assert isinstance(subrng, JRNG)
        assert subrng.key == subkey


def test_integers():
    rng = JRNG(42)
    key = rng.key
    rvs = rng.integers(0, 10, 10000)
    assert rng.key != key
    assert rvs.shape == (10000,)
    assert rvs.min() == 0
    assert rvs.max() == 9


def test_random():
    rng = JRNG(42)
    key = rng.key
    rvs = rng.random(10000)
    assert rng.key != key
    assert rvs.shape == (10000,)
    assert rvs.min() >= 0.0
    assert rvs.max() < 1.0


def test_choice():
    rng = JRNG(42)
    key = rng.key
    a = jnp.array([1, 2, 3])
    rvs = rng.choice(a, 10000)
    assert rng.key != key
    assert rvs.shape == (10000,)
    assert (jnp.unique(rvs) == a).all()


def test_bytes():
    rng = JRNG(42)
    key = rng.key
    rvs = rng.bytes(12)
    assert rng.key != key
    assert isinstance(rvs, bytes)
    assert len(rvs) == 12


def test_permutation():
    rng = JRNG(42)
    key = rng.key
    rvs = rng.permutation(100)
    assert rng.key != key
    assert (jnp.unique(rvs) == jnp.arange(100)).all()
