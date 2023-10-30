"""
The utility function for manipulating ellipsoids
"""

import numpy as np

import pydrake.solvers as solvers
import pydrake.symbolic as sym


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
