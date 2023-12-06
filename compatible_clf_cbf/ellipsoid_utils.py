"""
The utility function for manipulating ellipsoids
"""

from typing import Tuple

import numpy as np

import pydrake.solvers as solvers
import pydrake.symbolic as sym

import compatible_clf_cbf.utils as utils


def add_max_volume_linear_cost(
    prog: solvers.MathematicalProgram,
    S: np.ndarray,
    b: np.ndarray,
    c: sym.Variable,
    S_bar: np.ndarray,
    b_bar: np.ndarray,
    c_bar: float,
) -> solvers.Binding[solvers.LinearCost]:
    """
    Adds the linear cost as the linearization of an objective which correlates
    to the volume of the ellipsoid ℰ={x | xᵀSx+bᵀx+c≤0}

    To maximize the ellipsoid voluem, we can maximize
    log(bᵀS⁻¹b/4-c) - 1/n * log(det(S))

    The linearization of this function at (S̅, b̅, c̅) is
    max trace(⌈c   bᵀ/2⌉ᵀ  ⌈c̅,   b̅ᵀ/2⌉⁻¹) -(1+1/n)*trace(Sᵀ * S̅⁻¹)
              ⌊b/2    S⌋   ⌊b̅/2     S̅⌋

    Check doc/maximize_inner_ellipsoid.md for more details.

    Args:
      prog: The optimization problem to which the cost is added.
      S: A symmetric matrix of decision variables. S must have been registered
        in `prog` already.
      b: A vector of decision variables. b must have been registered in `prog`
        already.
      c: A decision variable. c must have been registered in `prog` already.
      S_bar: A symmetric matrix of floats, where we linearize the objective.
      b_bar: A vector of floats, where we linearized the objective.
    """

    n = S.shape[0]
    assert S.shape == (n, n)
    assert b.shape == (n,)
    assert S_bar.shape == (n, n)
    assert b_bar.shape == (n,)
    mat = np.empty((n + 1, n + 1), dtype=object)
    mat[0, 0] = c
    mat[0, 1:] = b / 2
    mat[1:, 0] = b / 2
    mat[1:, 1:] = S

    mat_bar = np.empty((n + 1, n + 1))
    mat_bar[0, 0] = c_bar
    mat_bar[0, 1:] = b_bar / 2
    mat_bar[1:, 0] = b_bar / 2
    mat_bar[1:, 1:] = S_bar
    mat_bar_inv = np.linalg.inv(mat_bar)
    S_bar_inv = np.linalg.inv(S_bar)

    cost_expr = np.trace(mat.T @ mat_bar_inv) - (1 + 1.0 / n) * np.trace(
        S.T @ S_bar_inv
    )
    cost = prog.AddLinearCost(-cost_expr)
    return cost


def maximize_inner_ellipsoid_sequentially(
    prog: solvers.MathematicalProgram,
    S: np.ndarray,
    b: np.ndarray,
    c: sym.Variable,
    S_init: np.ndarray,
    b_init: np.ndarray,
    c_init: float,
    max_iter: int = 10,
    convergence_tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Maximize the inner ellipsoid ℰ={x | xᵀSx+bᵀx+c≤0} through a sequence of
    convex programs.

    Args:
      prog: The optimization program that already contains all the constraints
        that the ellipsoid is inside the set.
      S: A symmetric matrix of decision variables. S must have been registered
        in `prog` already.
      b: A vector of decision variables. b must have been registered in `prog`
        already.
      c: A decision variable. c must have been registered in `prog` already.
      S_init: A symmetric matrix of floats, the initial guess of S.
      b_init: A vector of floats, the initial guess of b.
      c_init: A float, the initial guess of c.
    """
    S_bar = S_init
    b_bar = b_init
    c_bar = c_init

    def volume(S: np.ndarray, b: np.ndarray, c: float) -> float:
        n = S.shape[0]
        return (b.dot(np.linalg.solve(S, b)) / 4 - c) ** (n / 2) / np.sqrt(
            np.linalg.det(S)
        )

    cost = None
    volume_prev = volume(S_init, b_init, c_init)
    assert max_iter >= 1
    for i in range(max_iter):
        if cost is not None:
            prog.RemoveCost(cost)
        cost = add_max_volume_linear_cost(prog, S, b, c, S_bar, b_bar, c_bar)
        # The center of the ellipsoid {x | xᵀS_bar*x+b_barᵀx+c_bar≤0}
        ellipsoid_center = -np.linalg.solve(S_bar, b_bar) / 2
        # To constrain that {x | xᵀSx+bᵀx+c≤0} is a valid ellipsoid, we want it
        # to contain at least one point. We choose that contained point to be
        # ellipsoid_center.
        prog.AddLinearConstraint(
            ellipsoid_center.dot(S @ ellipsoid_center) + np.dot(b, ellipsoid_center) + c
            <= 0
        )
        result = solvers.Solve(prog)
        assert result.is_success()
        S_result = result.GetSolution(S)
        b_result = result.GetSolution(b)
        c_result = result.GetSolution(c)
        volume_result = volume(S_result, b_result, c_result)
        if volume_result - volume_prev <= convergence_tol:
            break
        else:
            volume_prev = volume_result
            S_bar = S_result
            b_bar = b_result
            c_bar = c_result
    return S_result, b_result, c_result


def add_minimize_ellipsoid_volume(
    prog: solvers.MathematicalProgram, S: np.ndarray, b: np.ndarray, c: sym.Variable
) -> sym.Variable:
    """
    Minimize the volume of the ellipsoid {x | xᵀSx + bᵀx + c ≤ 0}
    where S, b, and c are decision variables.

    See doc/minimize_ellipsoid_volume.md for the details (you will need to
    enable MathJax in your markdown viewer).

    We minimize the volume through the convex program
    min r
    s.t ⌈c+r  bᵀ/2⌉ is psd
        ⌊b/2     S⌋

        log det(S) >= 0

    Args:
      S: a symmetric matrix of decision variables. S must have been registered
      in `prog` already. It is the user's responsibility to impose "S is psd".
      b: a vector of decision variables. b must have been registered in `prog` already.
      c: a symbolic Variable. c must have been registered in `prog` already.
    Returns:
      r: The slack decision variable.
    """
    x_dim = S.shape[0]
    assert S.shape == (x_dim, x_dim)
    r = prog.NewContinuousVariables(1, "r")[0]
    prog.AddLinearCost(r)
    psd_mat = np.empty((x_dim + 1, x_dim + 1), dtype=object)
    psd_mat[0, 0] = c + r
    psd_mat[0, 1:] = b.T / 2
    psd_mat[1:, 0] = b / 2
    psd_mat[1:, 1:] = S
    prog.AddPositiveSemidefiniteConstraint(psd_mat)
    utils.add_log_det_lower(prog, S, lower=0.0)
    return r
