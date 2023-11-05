import compatible_clf_cbf.ellipsoid_utils as mut

import jax.numpy as jnp
import jax
import numpy as np
import pytest  # noqa

import pydrake.solvers as solvers
import pydrake.symbolic as sym


def eval_max_volume_cost(S: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray):
    """
    Evaluate the cost
    log(bᵀS⁻¹b/4-c) - 1/n * log(det(S))
    """
    n = S.shape[0]
    return jnp.log(b.dot(jnp.linalg.solve(S, b)) / 4 - c) - 1.0 / n * jnp.log(
        jnp.linalg.det(S)
    )


def check_add_max_volume_linear_cost(
    cost: solvers.Binding[solvers.LinearCost],
    S: np.ndarray,
    b: np.ndarray,
    c: sym.Variable,
    S_val: np.ndarray,
    b_val: np.ndarray,
    c_val: float,
    S_bar: np.ndarray,
    b_bar: np.ndarray,
    c_bar: float,
):
    """
    Evaluate the `cost` term at (S, b, c) = (S_val, b_val, c_val). Make sure
    that the evaluated value is the same as taking the linearization of the
    nonlinear cost numerically through JAX.
    """
    S_bar_jnp = jnp.array(S_bar)
    b_bar_jnp = jnp.array(b_bar)
    S_grad, b_grad, c_grad = jax.grad(eval_max_volume_cost, argnums=(0, 1, 2))(
        S_bar_jnp, b_bar_jnp, c_bar
    )
    cost_val_expected = -(
        np.trace(S_grad.T @ S_val) + b_grad.dot(b_val) + c_grad * c_val
    )
    env = dict()
    for i in range(S.shape[0]):
        for j in range(i, S.shape[1]):
            env[S[i, j]] = S_val[i, j]
    for b_var, b_var_val in zip(b.flat, b_val.flat):
        env[b_var] = b_var_val
    env[c] = c_val
    cost_expr = cost.evaluator().a() @ cost.variables() + cost.evaluator().b()
    cost_val = cost_expr.Evaluate(env)
    np.testing.assert_allclose(cost_val, cost_val_expected)


def test_add_max_volume_linear_cost1():
    # Test add_max_volume_linear_cost with an odd dimension n = 3.
    n = 3
    prog = solvers.MathematicalProgram()
    S = prog.NewSymmetricContinuousVariables(n, "S")
    prog.AddPositiveSemidefiniteConstraint(S)
    b = prog.NewContinuousVariables(n, "b")
    c = prog.NewContinuousVariables(1, "c")[0]

    S_bar = np.diag(np.array([1, 2.0, 3.0]))
    b_bar = np.array([4.0, 5.0, 6.0])
    c_bar = -10.0

    cost = mut.add_max_volume_linear_cost(prog, S, b, c, S_bar, b_bar, c_bar)
    S_val = np.array([[6, 2, 3], [2, 10, 4], [3, 4, 10.0]])
    b_val = np.array([1.0, 2.0, 3.0])
    c_val = -5.0
    check_add_max_volume_linear_cost(
        cost, S, b, c, S_val, b_val, c_val, S_bar, b_bar, c_bar
    )


def test_add_max_volume_linear_cost2():
    # Test add_max_volume_linear_cost with an even dimension n = 4.
    n = 4
    prog = solvers.MathematicalProgram()
    S = prog.NewSymmetricContinuousVariables(n, "S")
    prog.AddPositiveSemidefiniteConstraint(S)
    b = prog.NewContinuousVariables(n, "b")
    c = prog.NewContinuousVariables(1, "c")[0]

    S_bar = np.diag(np.array([1, 2.0, 3.0, 4.0]))
    S_bar[3, 0] = 0.1
    S_bar[0, 3] = 0.1
    b_bar = np.array([4.0, 5.0, 6.0, 7.0])
    c_bar = -10.0

    cost = mut.add_max_volume_linear_cost(prog, S, b, c, S_bar, b_bar, c_bar)
    S_val = np.array([[6, 2, 3, 0], [2, 10, 4, 1], [3, 4, 10.0, 1], [1, 2, 3, 8]])
    S_val = (S_val + S_val.T) / 2
    b_val = np.array([1.0, 2.0, 3.0, 4.0])
    c_val = -5.0
    check_add_max_volume_linear_cost(
        cost, S, b, c, S_val, b_val, c_val, S_bar, b_bar, c_bar
    )
