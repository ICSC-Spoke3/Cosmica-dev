from jax import Array
from PyCosmica.structures import PropagationState, PropagationConstants, DiffusionTensor


def diffusion_tensor_symmetric(state: PropagationState, const: PropagationConstants, key: Array) -> DiffusionTensor:
    pass