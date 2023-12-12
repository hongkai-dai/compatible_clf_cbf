import dataclasses
from typing import Optional, Tuple, Union
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


def is_sos(
    poly: sym.Polynomial,
    solver_id: Optional[solvers.SolverId] = None,
    solver_options: Optional[solvers.SolverOptions] = None,
):
    prog = solvers.MathematicalProgram()
    prog.AddIndeterminates(poly.indeterminates())
    assert poly.decision_variables().empty()
    prog.AddSosConstraint(poly)
    if solver_id is None:
        result = solvers.Solve(prog, None, solver_options)
    else:
        solver = solvers.MakeSolver(solver_id)
        result = solver.Solve(prog, None, solver_options)
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


@dataclasses.dataclass
class ContainmentLagrangian:
    """
    To certify that an algebraic set { x | f(x) <= 0} is contained in another
    algebraic set {x | g(x) < 0}, we impose the condition
    -1 - ϕ₁(x))g(x) + ϕ₂(x)f(x) is sos
    ϕ₁(x) is sos, ϕ₂(x) is sos
    """

    # ϕ₂(x) in the documentation above.
    inner: sym.Polynomial
    # ϕ₁(x) in the documentation above.
    outer: sym.Polynomial

    def add_constraint(
        self, prog, inner_poly: sym.Polynomial, outer_poly: sym.Polynomial
    ) -> Tuple[sym.Polynomial, np.ndarray]:
        return prog.AddSosConstraint(
            -1 - self.outer * outer_poly + self.inner * inner_poly
        )


@dataclasses.dataclass
class ContainmentLagrangianDegree:
    """
    The degree of the polynomials in ContainmentLagrangian
    If degree < 0, then the Lagrangian polynomial is 1.
    """

    inner: int = -1
    outer: int = -1

    def construct_lagrangian(
        self, prog: solvers.MathematicalProgram, x: sym.Variables
    ) -> ContainmentLagrangian:
        if self.inner < 0:
            inner_lagrangian = sym.Polynomial(1)
        elif self.inner == 0:
            inner_lagrangian_var = prog.NewContinuousVariables(1)[0]
            prog.AddBoundingBoxConstraint(0, np.inf, inner_lagrangian_var)
            inner_lagrangian = sym.Polynomial(
                {sym.Monomial(): sym.Expression(inner_lagrangian_var)}
            )
        else:
            inner_lagrangian, _ = prog.NewSosPolynomial(x, self.inner)
        if self.outer < 0:
            outer_lagrangian = sym.Polynomial(1)
        elif self.outer == 0:
            outer_lagrangian_var = prog.NewContinuousVariables(1)[0]
            prog.AddBoundingBoxConstraint(0, np.inf, outer_lagrangian_var)
            outer_lagrangian = sym.Polynomial(
                {sym.Monomial(): sym.Expression(outer_lagrangian_var)}
            )
        else:
            outer_lagrangian, _ = prog.NewSosPolynomial(x, self.outer)
        return ContainmentLagrangian(inner=inner_lagrangian, outer=outer_lagrangian)


def to_lower_triangular_columns(mat: np.ndarray) -> np.ndarray:
    assert mat.shape[0] == mat.shape[1]
    dim = mat.shape[0]
    ret = np.empty(int((dim + 1) * dim / 2), dtype=mat.dtype)
    count = 0
    for col in range(dim):
        ret[count: count + dim - col] = mat[col:, col]
        count += dim - col
    return ret
