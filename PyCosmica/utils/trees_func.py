from jax import numpy as jnp, tree_map, vmap

def pytrees_stack(pytrees, axis=0):
    results = tree_map(
        lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results

def pytrees_static_stack(pytrees, axis=0):
    results = tree_map(
        lambda *values: tuple(map(float, jnp.stack(values, axis=axis))), *pytrees)
    return results

def pytrees_unstack(pytree, var=0, axis=0):
    return [tree_map(lambda x: x[i], pytree) for i in range(pytree[var].shape[axis])]

def pytrees_flatten(pytree):
    return tree_map(lambda x: x.flatten(), pytree)
