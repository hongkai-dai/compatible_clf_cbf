import dataclasses
from typing import Optional, Union
import numpy as np

import pydrake.symbolic as sym
import pydrake.solvers as solvers


def check_array_of_polynomials(p: np.ndarray, x_set: sym.Variables) -> None:
    """
    Check if each element of p is a symbolic polynomial, whose indeterminates
    are a subset of `x_set`.
    """
    assert isinstance(p, np.ndarray)
    assert isinstance(x_set, sym.Variables)

    for p_i in p.flat:
        assert isinstance(p_i, sym.Polynomial)
        if not p_i.indeterminates().IsSubsetOf(x_set):
            raise Exception(f"{p_i}'s indeterminates is not a subset of {x_set}")


def check_polynomial_arrays_equal(p: np.ndarray, q: np.ndarray, tol: float):
    assert p.shape == q.shape
    for p_i, q_i in zip(p.flat, q.flat):
        assert p_i.CoefficientsAlmostEqual(q_i, tol)


def get_polynomial_result(
    result: solvers.MathematicalProgramResult,
    p: Union[sym.Polynomial, np.ndarray],
    coefficient_tol: Optional[float] = None,
) -> Union[sym.Polynomial, np.ndarray]:
    """
    Given a MathematicalProgramResult and an array of symbolic Polynomials
    (or a single symbolic Polynomial), return the result of these symbolic
    Polynomials. Remove the terms in the polynomials if the absolute vlues of
    the coefficients are <= coefficient_tol.
    """
    if isinstance(p, sym.Polynomial):
        p_result = result.GetSolution(p)
        if coefficient_tol is not None:
            return p_result.RemoveTermsWithSmallCoefficients(coefficient_tol)
        else:
            return p_result
    else:
        p_result = np.array([result.GetSolution(p_i) for p_i in p.flat]).reshape(
            p.shape
        )
        if coefficient_tol is not None:
            p_result = np.array(
                [
                    p_result_i.RemoveTermsWithSmallCoefficients(coefficient_tol)
                    for p_result_i in p_result.flat
                ]
            ).reshape(p.shape)
        return p_result


def is_sos(p: sym.Polynomial) -> bool:
    prog = solvers.MathematicalProgram()
    prog.AddIndeterminates(p.indeterminates())
    prog.AddSosConstraint(p)
    result = solvers.Solve(prog)
    return result.is_success()


@dataclasses.dataclass
class LogDetLowerRet:
    Z: np.ndarray
    t: np.ndarray


def add_log_det_lower(
    prog: solvers.MathematicalProgram, X: np.ndarray, lower: float
) -> LogDetLowerRet:
    """
    Impose the constraint that log(det(X)) >= lower where X is a psd matrix.

    This can be formulated through semidefinite and exponential cone
    constraints. We introduce slack variable t, and a lower-diagonal matrix Z,
    with the constraint

        ⌈X         Z⌉ is positive semidifinite.
        ⌊Zᵀ  diag(Z)⌋

        log(Z(i, i)) >= t(i)
        ∑ᵢt(i) >= lower

    TODO(hongkai.dai): move this function into Drake.
    """
    if lower < 0:
        raise Warning(
            "add_log_det_lower(): lower is negative. You do not need this constraint."
        )
    X_rows = X.shape[0]
    assert X.shape == (X_rows, X_rows)
    Z_lower = prog.NewContinuousVariables(int(X_rows * (X_rows + 1) / 2))
    Z = np.empty((X_rows, X_rows), dtype=object)
    Z_lower_count = 0
    for i in range(X_rows):
        for j in range(i + 1):
            Z[i, j] = Z_lower[Z_lower_count]
            Z_lower_count += 1
        for j in range(i + 1, X_rows):
            Z[i, j] = 0
    t = prog.NewContinuousVariables(X_rows)

    psd_mat = np.zeros((X_rows * 2, X_rows * 2), dtype=object)
    psd_mat[:X_rows, :X_rows] = X
    psd_mat[:X_rows, X_rows:] = Z
    psd_mat[X_rows:, :X_rows] = Z.T
    psd_mat[X_rows:, X_rows:] = np.diagonal(Z)
    prog.AddPositiveSemidefiniteConstraint(psd_mat)

    for i in range(X_rows):
        prog.AddExponentialConeConstraint(np.array([Z[i, i], 1, t[i]]))
    prog.AddLinearConstraint(np.ones((1, X_rows)), lower, np.inf, t)

    return LogDetLowerRet(Z=Z, t=t)


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
    add_log_det_lower(prog, S, lower=0.0)
    return r
