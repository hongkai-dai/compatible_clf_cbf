import compatible_clf_cbf.ellipsoid_utils as mut

import jax.numpy as jnp
import jax
import numpy as np
import pytest  # noqa

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.math


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


def check_maximize_inner_ellipsoid_sequentially(
    box_size: np.ndarray, box_center: np.ndarray, box_rotation: np.ndarray
):
    """
    We find the maximal ellipsoid contained within a box
    { x | x = R * x_bar + p, -box_size/2 <= x_bar <= box_size/2}.
    """
    dim = box_size.size
    assert box_center.shape == (dim,)
    assert box_rotation.shape == (dim, dim)

    # The box can be written as
    # -box_size/2 <= R.T * (x-p) <= box_size/2
    # Let's denote this "box constraint" as C * x <= d.
    # According to s-lemma, an ellipsoid
    # {x | x.T*S*x + b.T* x + c<= 0} is inside the ellipsoid if and only if
    # xᵀSx+bᵀx+c − γ(Cᵢx−dᵢ) for some γ>= 0
    # Namely the matrix
    # [S             b/2 - γCᵢ] is psd.
    # [(b/2 - γCᵢ).T   c + γdᵢ]
    prog = solvers.MathematicalProgram()
    S = prog.NewSymmetricContinuousVariables(dim, "S")
    b = prog.NewContinuousVariables(dim, "b")
    c = prog.NewContinuousVariables(1, "c")[0]
    prog.AddPositiveSemidefiniteConstraint(S)
    gamma = prog.NewContinuousVariables(2 * dim)
    prog.AddBoundingBoxConstraint(0, np.inf, gamma)
    C = np.concatenate((box_rotation.T, -box_rotation.T), axis=0)
    d = np.concatenate(
        (
            box_size / 2 + box_rotation.T @ box_center,
            box_size / 2 - box_rotation.T @ box_center,
        )
    )
    for i in range(2 * dim):
        psd_mat = np.empty((dim + 1, dim + 1), dtype=object)
        psd_mat[:dim, :dim] = S
        psd_mat[:dim, -1] = b / 2 - gamma[i] * C[i, :]
        psd_mat[-1, :dim] = b / 2 - gamma[i] * C[i, :]
        psd_mat[-1, -1] = c + gamma[i] * d[i]
        prog.AddPositiveSemidefiniteConstraint(psd_mat)

    S_init = np.eye(dim)
    b_init = -2 * box_center
    c_init = box_center.dot(box_center) - 0.99 * box_size.min() ** 2

    S_sol, b_sol, c_sol = mut.maximize_inner_ellipsoid_sequentially(
        prog, S, b, c, S_init, b_init, c_init, max_iter=10, convergence_tol=1e-5
    )
    # The maximal inner ellipsoid is the elliposoid
    # {R * y + p | y.T * diag(1 / box_size**2) * y <= 1}.
    S_max = box_rotation.T @ np.diag(1 / box_size**2) @ box_rotation
    b_max = -2 * S_max @ box_center
    c_max = box_center.dot(S_max @ box_center) - 1

    ratio = c_sol / c_max
    np.testing.assert_allclose(S_sol, S_max * ratio, atol=1e-5)
    np.testing.assert_allclose(b_sol, b_max * ratio, atol=1e-5)


def test_maximize_inner_ellipsoid_sequentially():
    # 2D case, axis aligned box.
    check_maximize_inner_ellipsoid_sequentially(
        np.array([1.0, 2.0]), np.array([0.5, 0.6]), np.eye(2)
    )

    # 2D case, rotated box
    check_maximize_inner_ellipsoid_sequentially(
        np.array([2.0, 4.0]),
        np.array([0.5, 1.0]),
        np.array([[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]]),
    )

    # 3D case, rotated box.
    check_maximize_inner_ellipsoid_sequentially(
        np.array([2, 3, 5.0]),
        np.array([0.5, -0.2, 0.3]),
        pydrake.math.RotationMatrix(pydrake.math.RollPitchYaw(0.2, 0.3, 0.5)).matrix(),
    )


def test_add_ellipsoid_contain_pts_constraint():
    prog = solvers.MathematicalProgram()
    S = prog.NewSymmetricContinuousVariables(3, "S")
    prog.AddPositiveSemidefiniteConstraint(S)
    b = prog.NewContinuousVariables(3, "b")
    c = prog.NewContinuousVariables(1, "c")[0]
    pts = np.array([[1, 2, 3], [0.5, 1, -2]])
    constraint = mut.add_ellipsoid_contain_pts_constraint(prog, S, b, c, pts)
    result = solvers.Solve(prog)
    assert result.is_success()
    S_sol = result.GetSolution(S)
    b_sol = result.GetSolution(b)
    c_sol = result.GetSolution(c)
    assert np.all(np.linalg.eigvals(S_sol) >= 0)
    for i in range(pts.shape[0]):
        assert pts[i].dot(S_sol @ pts[i]) + b_sol.dot(pts[i]) + c_sol <= 0
    prog.RemoveConstraint(constraint)
    assert (
        len(prog.linear_constraints()) == 0
        and len(prog.linear_equality_constraints()) == 0
    )
