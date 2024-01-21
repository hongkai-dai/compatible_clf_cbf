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
                        sym.Monomial(x[0], 3): sym.Expression(1.0 / 6.0),
                    }
                ),
            ]
        )
        g = np.array([[sym.Polynomial(1)], [sym.Polynomial(-1)]])
    else:
        f = np.array([0, -x[0] + x[0] ** 3 / 6])
        g = np.array([[1], [-1]])

    return (f, g)


def affine_trig_poly_state_constraints(x: np.ndarray) -> sym.Polynomial:
    """
    With the state x̅ = [sinx₀, cosx₀−1, x₁], the state constraint is x̅₀² + (x̅₁+1)²=1
    """
    return sym.Polynomial(x[0] ** 2 + x[1] ** 2 + 2 * x[1])


def affine_trig_poly_dynamics(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    With the trigonometric state x̅ = [sinx₀, cosx₀−1, x₁], we can write the
    dynamics as affine form f(x̅) + g(x̅)u
    """
    assert x.shape == (3,)
    if x.dtype == object:
        f = np.array([sym.Polynomial(), sym.Polynomial(), sym.Polynomial(-x[0])])
        g = np.array(
            [[sym.Polynomial(x[1] + 1)], [sym.Polynomial(-x[0])], [sym.Polynomial(-1)]]
        )
    else:
        f = np.array([0, 0, -x[0]])
        g = np.array([[x[1] + 1], [-x[0]], [-1]])
    return f, g
