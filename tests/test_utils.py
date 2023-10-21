import compatible_clf_cbf.utils as mut

import numpy as np
import pytest  # noqa

import pydrake.symbolic as sym
import pydrake.solvers as solvers


def test_check_array_of_polynomials():
    x = sym.MakeVectorContinuousVariable(rows=3, name="x")
    x_set = sym.Variables(x)
    p = np.array([sym.Polynomial(x[0] * x[0]), sym.Polynomial(x[1] + 2)])
    mut.check_array_of_polynomials(p, x_set)


def check_psd(X: np.ndarray, tol: float):
    assert np.all((np.linalg.eig(X)[0] >= -tol))


def test_add_log_det_lower():
    def tester(lower):
        prog = solvers.MathematicalProgram()
        X = prog.NewSymmetricContinuousVariables(3, "X")
        ret = mut.add_log_det_lower(prog, X, lower)
        result = solvers.Solve(prog)
        assert result.is_success()
        X_sol = result.GetSolution(X)
        check_psd(X_sol, tol=1e-6)
        assert np.log(np.linalg.det(X_sol)) >= lower - 1e-6
        t_sol = result.GetSolution(ret.t)
        assert np.sum(t_sol) >= lower - 1E-6

    tester(2.0)
    tester(3.0)


def check_inscribed_ellipsoid(S_inner: np.ndarray, b_inner: np.ndarray, c_inner: float):
    """
    Find the largest ellipsoid {x | xᵀSx + bᵀx + c ≤ 0} contained within the
    inner ellipsoid {x | xᵀS_inner*x + b_innerᵀx + c_outer ≤ 0}. Obviously the
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


def test_add_maximize_ellipsoid_volume():
    check_inscribed_ellipsoid(S_inner=np.eye(2), b_inner=np.zeros(2), c_inner=-1)
    check_inscribed_ellipsoid(S_inner=np.eye(2), b_inner=np.array([1, 2]), c_inner=-1)
    check_inscribed_ellipsoid(
        S_inner=np.array([[1, 2], [2, 9]]), b_inner=np.array([1, 2]), c_inner=-1
    )
