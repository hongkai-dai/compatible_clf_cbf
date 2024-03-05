import dataclasses
from typing import List, Optional, Tuple, Union
from typing_extensions import Self
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
    To certify that an algebraic set { x | f(x) <= 0, g(x)=0} is contained in another
    algebraic set {x | h(x) <= 0}, we impose the condition
    -(1 + ϕ₁(x))h(x) + ϕ₂(x)f(x) + ϕ₃(x)*g(x) is sos
    ϕ₁(x) is sos, ϕ₂(x) is sos
    """

    # ϕ₂(x) in the documentation above, an array of polynomials.
    inner_ineq: np.ndarray
    # ϕ₃(x) in the documentation above, an array of polynomials.
    inner_eq: np.ndarray
    # ϕ₁(x) in the documentation above.
    outer: sym.Polynomial

    def add_constraint(
        self,
        prog,
        inner_ineq_poly: np.ndarray,
        inner_eq_poly: Optional[np.ndarray],
        outer_poly: sym.Polynomial,
    ) -> Tuple[sym.Polynomial, np.ndarray]:
        sos_condition = -(1 + self.outer) * outer_poly + self.inner_ineq.dot(
            inner_ineq_poly
        )
        if inner_eq_poly is None:
            assert self.inner_eq is None or self.inner_eq.size == 0
        else:
            sos_condition += self.inner_eq.dot(inner_eq_poly)
        return prog.AddSosConstraint(sos_condition)

    def get_result(self, result: solvers.MathematicalProgramResult) -> Self:
        return ContainmentLagrangian(
            inner_ineq=get_polynomial_result(result, self.inner_ineq),
            inner_eq=get_polynomial_result(result, self.inner_eq),
            outer=get_polynomial_result(result, self.outer),
        )


@dataclasses.dataclass
class ContainmentLagrangianDegree:
    """
    The degree of the polynomials in ContainmentLagrangian
    If degree < 0, then the Lagrangian polynomial is 1.
    """

    inner_ineq: List[int]
    inner_eq: List[int]
    outer: int

    def construct_lagrangian(
        self, prog: solvers.MathematicalProgram, x: sym.Variables
    ) -> ContainmentLagrangian:
        inner_ineq_lagrangians = [None] * len(self.inner_ineq)
        inner_eq_lagrangians = [None] * len(self.inner_eq)
        for i, inner_ineq_i in enumerate(self.inner_ineq):
            if inner_ineq_i < 0:
                inner_ineq_lagrangians[i] = sym.Polynomial(1)
            else:
                inner_ineq_lagrangians[i] = new_sos_polynomial(prog, x, inner_ineq_i)[0]
        inner_ineq_lagrangians = np.array(inner_ineq_lagrangians)

        for i, inner_eq_i in enumerate(self.inner_eq):
            if inner_eq_i < 0:
                raise Exception(
                    f"inner_eq[{i}] = {inner_eq_i}, should be non-negative."
                )
            else:
                inner_eq_lagrangians[i] = prog.NewFreePolynomial(x, inner_eq_i)
        inner_eq_lagrangians = np.array(inner_eq_lagrangians)
        if self.outer < 0:
            # The Lagrangian multiply with the outer set is
            # (1 + outer_lagrangian), hence we can set outer_lagrangian = 0,
            # such that the multiplied Lagrangian is 1.
            outer_lagrangian = sym.Polynomial(0)
        else:
            outer_lagrangian = new_sos_polynomial(prog, x, self.outer)[0]
        return ContainmentLagrangian(
            inner_ineq=inner_ineq_lagrangians,
            inner_eq=inner_eq_lagrangians,
            outer=outer_lagrangian,
        )


def to_lower_triangular_columns(mat: np.ndarray) -> np.ndarray:
    assert mat.shape[0] == mat.shape[1]
    dim = mat.shape[0]
    ret = np.empty(int((dim + 1) * dim / 2), dtype=mat.dtype)
    count = 0
    for col in range(dim):
        ret[count : count + dim - col] = mat[col:, col]  # noqa
        count += dim - col
    return ret


def solve_with_id(
    prog: solvers.MathematicalProgram,
    solver_id: Optional[solvers.SolverId] = None,
    solver_options: Optional[solvers.SolverOptions] = None,
    backoff_scale: Optional[float] = None,
) -> solvers.MathematicalProgramResult:
    """
    Args:
      backoff_scale: when solving an optimization problem with an objective function,
      we first solve the problem to optimality, and then "back off" a little bit to find
      a sub-optimal but strictly feasible solution. backoff_scale=0 corresponds to no
      backoff. Note that during backing off, we will modify the original `prog`.
    """
    if solver_id is None:
        result = solvers.Solve(prog, None, solver_options)
    else:
        solver = solvers.MakeSolver(solver_id)
        result = solver.Solve(prog, None, solver_options)
    if (
        len(prog.linear_costs()) > 0 or len(prog.quadratic_costs()) > 0
    ) and backoff_scale is not None:
        assert (
            len(prog.linear_costs()) == 1
        ), "TODO(hongkai.dai): support program with multiple LinearCost objects."
        assert (
            len(prog.quadratic_costs()) == 0
        ), "TODO(hongkai.dai): we currently only support program with linear costs."
        assert backoff_scale >= 0, "backoff_scale should be non-negative."
        optimal_cost = result.get_optimal_cost()
        coeff_cost = prog.linear_costs()[0].evaluator().a()
        var_cost = prog.linear_costs()[0].variables()
        constant_cost = prog.linear_costs()[0].evaluator().b()
        prog.RemoveCost(prog.linear_costs()[0])
        cost_upper_bound = (
            optimal_cost * (1 + backoff_scale)
            if optimal_cost > 0
            else optimal_cost * (1 - backoff_scale)
        )
        prog.AddLinearConstraint(
            coeff_cost, -np.inf, cost_upper_bound - constant_cost, var_cost
        )
        if solver_id is None:
            result = solvers.Solve(prog, None, solver_options)
        else:
            result = solver.Solve(prog, None, solver_options)
    return result


def new_sos_polynomial(
    prog: solvers.MathematicalProgram,
    x_set: sym.Variables,
    degree: int,
    zero_at_origin: bool = False,
) -> Tuple[sym.Polynomial, np.ndarray]:
    """
    Returns a new SOS polynomial (where the coefficients are decision variables).

    Args:
      zero_at_origin: where this SOS polynomial sos_poly(0) = 0.

    Return:
      sos_poly: The newly constructed sos polynomial.
      gram: The Gram matrix of sos_poly.
    """
    if degree == 0:
        if zero_at_origin:
            # This polynomial is a constant 0.
            return sym.Polynomial(), np.array([[]])
        else:
            coeff = prog.NewContinuousVariables(1)[0]
            prog.AddBoundingBoxConstraint(0, np.inf, coeff)
            sos_poly = sym.Polynomial({sym.Monomial(): sym.Expression(coeff)})
            return sos_poly, np.array([[coeff]])
    else:
        if zero_at_origin:
            # This polynomial cannot have constant or linear terms.
            monomial_basis = sym.MonomialBasis(x_set, int(np.floor(degree / 2)))
            assert monomial_basis[-1].total_degree() == 0
            sos_poly, gram = prog.NewSosPolynomial(monomial_basis[:-1])
        else:
            sos_poly, gram = prog.NewSosPolynomial(x_set, degree)
        return sos_poly, gram


@dataclasses.dataclass
class BinarySearchOptions:
    min: float
    max: float
    tol: float

    def check(self):
        assert self.min <= self.max
        assert self.tol > 0
