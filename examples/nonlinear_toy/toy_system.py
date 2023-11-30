from typing import Tuple

import numpy as np
import pydrake.symbolic as sym


def affine_dynamics(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the dynamics
    ẋ₀ = u
    ẋ₁ = −x₀ + 1/6x₀³−u
    as ẋ = f(x) + g(x)*u

    return f and g.
    """
    assert x.shape == (2,)
    if x.dtype == object:
        f = np.array(
            [
                sym.Polynomial(),
                sym.Polynomial(
                    {
                        sym.Monomial(x[0]): sym.Expression(-1),
                        sym.Monomial(x[0], 3): sym.Expression(-1.0 / 6.0),
                    }
                ),
            ]
        )
        g = np.array([[sym.Polynomial(1)], [sym.Polynomial(-1)]])
    else:
        f = np.array([0, -x[0] + x[0] ** 3 / 6])
        g = np.array([[1], [-1]])

    return (f, g)
