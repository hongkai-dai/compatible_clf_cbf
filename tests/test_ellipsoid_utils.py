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


def check_psd(X: np.ndarray, tol: float):
    assert np.all((np.linalg.eig(X)[0] >= -tol))


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


def check_outer_ellipsoid(S_inner: np.ndarray, b_inner: np.ndarray, c_inner: float):
    """
    Find the smallest ellipsoid {x | xᵀSx + bᵀx + c ≤ 0} containing the
    inner ellipsoid {x | xᵀS_inner*x + b_innerᵀx + c_inner ≤ 0}. Obviously the
    largest ellipsoid should be the inner ellipsoid itself.
    """
    assert np.all((np.linalg.eig(S_inner)[0] > 0))
    x_dim = S_inner.shape[0]
    prog = solvers.MathematicalProgram()
    S = prog.NewSymmetricContinuousVariables(x_dim, "S")
    prog.AddPositiveSemidefiniteConstraint(S)
    b = prog.NewContinuousVariables(x_dim, "b")
    c = prog.NewContinuousVariables(1, "c")[0]

    r = mut.add_minimize_ellipsoid_volume(prog, S, b, c)

    # According to s-lemma, xᵀS_inner*x + b_innerᵀx + c_inner <= 0 implies
    # xᵀSx + bᵀx + c <= 0 iff there exists λ ≥ 0, such that
    # -(xᵀSx + bᵀx + c) + λ*(xᵀS_inner*x + b_innerᵀx + c_inner) >= 0 for all x.
    # Namely
    # ⌈ λS_inner - S   (λb_inner-b)/2⌉ is psd.
    # ⌊(λb_inner-b)ᵀ/2     λc_inner-c⌋
    lambda_var = prog.NewContinuousVariables(1, "lambda")[0]
    prog.AddBoundingBoxConstraint(0, np.inf, lambda_var)
    psd_mat = np.empty((x_dim + 1, x_dim + 1), dtype=object)
    psd_mat[:x_dim, :x_dim] = lambda_var * S_inner - S
    psd_mat[:x_dim, -1] = (lambda_var * b_inner - b) / 2
    psd_mat[-1, :x_dim] = (lambda_var * b_inner - b).T / 2
    psd_mat[-1, -1] = lambda_var * c_inner - c
    prog.AddPositiveSemidefiniteConstraint(psd_mat)
    result = solvers.Solve(prog)
    assert result.is_success
    S_sol = result.GetSolution(S)
    b_sol = result.GetSolution(b)
    c_sol = result.GetSolution(c)
    r_sol = result.GetSolution(r)
    np.testing.assert_allclose(
        r_sol, b_sol.dot(np.linalg.solve(S_sol, b_sol)) / 4 - c_sol
    )

    mat = np.empty((x_dim + 1, x_dim + 1))
    mat[0, 0] = c_sol + r_sol
    mat[0, 1:] = b_sol.T / 2
    mat[1:, 0] = b_sol / 2
    mat[1:, 1:] = S_sol
    check_psd(mat, tol=1e-6)

    # Make sure the (S, b, c) is a scaled version of
    # (S_inner, b_inner, c_inner), namely they correspond to the same
    # ellipsoid.
    factor = c_sol / c_inner
    np.testing.assert_allclose(S_sol, S_inner * factor, atol=1e-7)
    np.testing.assert_allclose(b_sol, b_inner * factor, atol=1e-7)


def test_add_minimize_ellipsoid_volume():
    check_outer_ellipsoid(S_inner=np.eye(2), b_inner=np.zeros(2), c_inner=-1)
    check_outer_ellipsoid(S_inner=np.eye(2), b_inner=np.array([1, 2]), c_inner=-1)
    check_outer_ellipsoid(
        S_inner=np.array([[1, 2], [2, 9]]), b_inner=np.array([1, 2]), c_inner=-1
    )
