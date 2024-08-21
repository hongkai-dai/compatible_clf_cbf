"""
The power converter system described in
"Advanced safety filter based on SOS control barrier and Lyapunov functions"
by M.Schneeberger, S.Mastellone and F.Dorfler
"""

from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np

import pydrake.symbolic as sym


def dynamics(
    x: Union[np.ndarray, jnp.ndarray], u: Union[np.ndarray, jnp.ndarray]
) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(x, np.ndarray):
        numpy_type = np
    elif isinstance(x, jnp.ndarray):
        numpy_type = jnp
    f = numpy_type.array(
        [
            -0.05 * x[0] - 57.9 * x[1] + 0.00919 * x[2],
            1710 * x[0] + 314 * x[2],
            -0.271 * x[0] - 314 * x[1],
        ]
    )
    g = numpy_type.array(
        [
            [0.05 - 57.9 * x[1], -57.9 * x[2]],
            [1710 + 1710 * x[0], 0],
            [0, 1710 + 1710 * x[0]],
        ]
    )
    return f + g @ u


def affine_dynamics(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f = np.array(
        [
            sym.Polynomial(-0.05 * x[0] - 57.9 * x[1] + 0.00919 * x[2]),
            sym.Polynomial(1710 * x[0] + 314 * x[2]),
            sym.Polynomial(-0.271 * x[0] - 314 * x[1]),
        ]
    )
    g = np.array(
        [
            [sym.Polynomial(0.05 - 57.9 * x[1]), sym.Polynomial(-57.9 * x[2])],
            [sym.Polynomial(1710 + 1710 * x[0]), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(1710 + 1710 * x[0])],
        ]
    )
    return f, g


def linearize_dynamics(x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.array(
        [
            [-0.05, -57.9 - 57.9 * u[0], 0.00919 - 57.9 * u[1]],
            [1710 + 1710 * u[0], 0, 314],
            [-0.271 + 1710 * u[1], -314, 0],
        ]
    )
    B = np.array(
        [
            [0.05 - 57.9 * x[1], -57.9 * x[2]],
            [1710 + 1710 * x[0], 0],
            [0, 1710 + 1710 * x[0]],
        ]
    )
    return A, B
