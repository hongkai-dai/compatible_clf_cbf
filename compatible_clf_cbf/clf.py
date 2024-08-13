"""
Certify and search Control Lyapunov function (CLF) through sum-of-squares optimization.
"""

from dataclasses import dataclass
import os
import pickle
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.autodiffutils

from compatible_clf_cbf.utils import (
    check_array_of_polynomials,
    check_polynomials_pass_origin,
    find_no_linear_term_variables,
    get_polynomial_result,
    new_free_polynomial_pass_origin,
    new_sos_polynomial,
    solve_with_id,
)
import compatible_clf_cbf.utils


@dataclass
class ClfWoInputLimitLagrangian:
    """
    The Lagrangians for proving the CLF condition for systems _without_ input limits.
    """

    # The Lagrangian λ₀(x) in (1+λ₀(x))*(∂V/∂x*f(x)+κ*V(x))
    dVdx_times_f: sym.Polynomial
    # The array of Lagrangians λ₁(x) in λ₁(x)*∂V/∂x*g(x)
    dVdx_times_g: np.ndarray
    # The Lagrangian for ρ − V(x)
    rho_minus_V: sym.Polynomial
    # The array of Lagrangians for state equality constraints.
    state_eq_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        dVdx_times_f_result = get_polynomial_result(
            result, self.dVdx_times_f, coefficient_tol
        )
        dVdx_times_g_result = get_polynomial_result(
            result, self.dVdx_times_g, coefficient_tol
        )
        rho_minus_V_result = get_polynomial_result(
            result, self.rho_minus_V, coefficient_tol
        )
        state_eq_constraints_result = (
            None
            if self.state_eq_constraints is None
            else get_polynomial_result(
                result, self.state_eq_constraints, coefficient_tol
            )
        )
        return ClfWoInputLimitLagrangian(
            dVdx_times_f=dVdx_times_f_result,
            dVdx_times_g=dVdx_times_g_result,
            rho_minus_V=rho_minus_V_result,
            state_eq_constraints=state_eq_constraints_result,
        )


@dataclass
class ClfWoInputLimitLagrangianDegrees:
    dVdx_times_f: int
    dVdx_times_g: List[int]
    rho_minus_V: int
    state_eq_constraints: Optional[List[int]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        x_equilibrium: np.ndarray,
    ) -> ClfWoInputLimitLagrangian:
        """
        Constructs the Lagrangians as SOS polynomials.
        """
        dVdx_times_f, _ = new_sos_polynomial(prog, x_set, self.dVdx_times_f)
        dVdx_times_g = np.array(
            [prog.NewFreePolynomial(x_set, degree) for degree in self.dVdx_times_g]
        )
        # We know that V(x_equilibrium) = 0 and dVdx at x_equilibrium is also
        # 0. Hence by
        # -(1+λ₀(x))*(∂V/∂x*f(x)+κ*V(x))−λ₁(x)*∂V/∂x*g(x)−λ₂(x)*(ρ−V(x))
        # being sos, we know that λ₂(x_equilibrium) = 0.
        # When x_equilibrium = 0, this means that λ₂(0) = 0. Since λ₂(x) is
        # sos, we know that its monomial basis doesn't contain 1.
        rho_minus_V, _ = new_sos_polynomial(
            prog, x_set, self.rho_minus_V, bool(np.all(x_equilibrium == 0))
        )

        state_eq_constraints = (
            None
            if self.state_eq_constraints is None
            else np.array(
                [
                    prog.NewFreePolynomial(x_set, degree)
                    for degree in self.state_eq_constraints
                ]
            )
        )
        return ClfWoInputLimitLagrangian(
            dVdx_times_f, dVdx_times_g, rho_minus_V, state_eq_constraints
        )


@dataclass
class ClfWInputLimitLagrangian:
    """
    The Lagrangians for proving the CLF condition for systems _with_ input limits.
    """

    # The Lagrangian λ₀(x) in (1+λ₀(x))(V(x)−ρ)xᵀx
    V_minus_rho: sym.Polynomial
    # The Lagrangians λᵢ(x) in ∑ᵢ λᵢ(x)*(∂V/∂x*(f(x)+g(x)uᵢ)+κ*V(x))
    Vdot: np.ndarray
    # The Lagrangians for state equality constraints
    state_eq_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        V_minus_rho_result: sym.Polynomial = get_polynomial_result(
            result, self.V_minus_rho, coefficient_tol
        )
        Vdot_result: np.ndarray = get_polynomial_result(
            result, self.Vdot, coefficient_tol
        )
        state_eq_constraints_result = (
            None
            if self.state_eq_constraints is None
            else get_polynomial_result(
                result, self.state_eq_constraints, coefficient_tol
            )
        )
        return ClfWInputLimitLagrangian(
            V_minus_rho=V_minus_rho_result,
            Vdot=Vdot_result,
            state_eq_constraints=state_eq_constraints_result,
        )


@dataclass
class ClfWInputLimitLagrangianDegrees:
    V_minus_rho: int
    Vdot: List[int]
    state_eq_constraints: Optional[List[int]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        x_equilibrium: np.ndarray,
    ) -> ClfWInputLimitLagrangian:
        V_minus_rho, _ = new_sos_polynomial(prog, x_set, self.V_minus_rho)
        Vdot = np.array(
            [new_sos_polynomial(prog, x_set, degree)[0] for degree in self.Vdot]
        )
        state_eq_constraints = (
            None
            if self.state_eq_constraints is None
            else np.array(
                [
                    prog.NewFreePolynomial(x_set, degree)
                    for degree in self.state_eq_constraints
                ]
            )
        )
        return ClfWInputLimitLagrangian(V_minus_rho, Vdot, state_eq_constraints)


class ControlLyapunov:
    """
    For a control affine system
    ẋ=f(x)+g(x)u, u∈𝒰
    we certify its CLF V, together with an inner approximation of the region of
    attraction (ROA) {x|V(x)≤ρ} through SOS.

    If the set 𝒰 is the entire space (namely the control input is not bounded),
    then V is an CLF iff the set satisfying the following conditions
    ∂V/∂x*f(x)+κ*V(x) > 0
    ∂V/∂x*g(x)=0
    V(x)≤ρ
    is empty.

    By positivestellasatz, this means that there exists λ₀(x), λ₁(x), λ₂(x)
    satisfying
    -(1+λ₀(x))*(∂V/∂x*f(x)+κ*V(x))−λ₁(x)*∂V/∂x*g(x)−λ₂(x)*(ρ−V(x)) is sos.
    λ₀(x), λ₂(x) are sos.

    If the set 𝒰 is a polytope with vertices u₁,..., uₙ, then V is an CLF iff
    (V(x)-ρ)xᵀx is always non-negative on the semi-algebraic set
    {x|∂V/∂x*(f(x)+g(x)uᵢ)≥ −κ*V(x), i=1,...,n}
    By positivestellasatz, we have
    (1+λ₀(x))(V(x)−ρ)xᵀx − ∑ᵢ λᵢ(x)*(∂V/∂x*(f(x)+g(x)uᵢ)+κ*V(x))
    λ₀(x), λᵢ(x) are sos.

    See Convex Synthesis and Verification of Control-Lyapunov and Barrier
    Functions with Input Constraints, Hongkai Dai and Frank Permenter, 2023.
    """

    def __init__(
        self,
        *,
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        x_equilibrium: np.ndarray,
        u_vertices: Optional[np.ndarray] = None,
        state_eq_constraints: Optional[np.ndarray] = None,
    ):
        """
        Args:
          f: np.ndarray
            An array of symbolic polynomials. The dynamics is ẋ = f(x)+g(x)u.
            The shape is (nx,)
          g: np.ndarray
            An array of symbolic polynomials. The dynamics is ẋ = f(x)+g(x)u.
            The shape is (nx, nu)
          x: np.ndarray
            An array of symbolic variables representing the state.
            The shape is (nx,)
          x_equilibrium: The equilibrium state. I strongly recommend using 0
            as the equilibrium state.
          u_vertices: The vertices of the input constraint polytope 𝒰. Each row
            is a vertex. If u_vertices=None, then the input is unconstrained.
          state_eq_constraints: An array of polynomials. Some dynamical systems
            have equality constraints on its states. For example, when the
            state include sinθ and cosθ (so that the dynamics is a polynomial
            function of state), we need to impose the equality constraint
            sin²θ+cos²θ=1 on the state. state_eq_constraints[i] = 0 is an
            equality constraint on the state.

        """
        assert len(f.shape) == 1
        assert len(g.shape) == 2
        self.nx: int = f.shape[0]
        self.nu: int = g.shape[1]
        assert g.shape == (self.nx, self.nu)
        assert x.shape == (self.nx,)
        self.f = f
        self.g = g
        self.x = x
        self.x_set: sym.Variables = sym.Variables(x)
        check_array_of_polynomials(f, self.x_set)
        check_array_of_polynomials(g, self.x_set)
        assert x_equilibrium.shape == (self.nx,)
        self.x_equilibrium = x_equilibrium
        if u_vertices is not None:
            assert u_vertices.shape[1] == self.nu
        self.u_vertices = u_vertices
        if state_eq_constraints is not None:
            check_array_of_polynomials(state_eq_constraints, self.x_set)
        self.state_eq_constraints = state_eq_constraints

    def search_lagrangian_given_clf(
        self,
        V: sym.Polynomial,
        rho: float,
        kappa: float,
        lagrangian_degrees: Union[
            ClfWInputLimitLagrangianDegrees, ClfWoInputLimitLagrangianDegrees
        ],
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
    ) -> Optional[Union[ClfWInputLimitLagrangian, ClfWoInputLimitLagrangian]]:
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x_set)
        lagrangians = lagrangian_degrees.to_lagrangians(
            prog, self.x_set, self.x_equilibrium
        )
        self._add_clf_condition(prog, V, lagrangians, rho, kappa)
        result = solve_with_id(prog, solver_id, solver_options)
        lagrangians_result = (
            lagrangians.get_result(result, lagrangian_coefficient_tol)
            if result.is_success()
            else None
        )
        return lagrangians_result

    def construct_search_clf_given_lagrangian(
        self,
        kappa: float,
        V_degree: int,
        lagrangians: Union[ClfWInputLimitLagrangian, ClfWoInputLimitLagrangian],
    ) -> Tuple[solvers.MathematicalProgram, sym.Polynomial, sym.Variable]:
        """
        Construct a mathematical program to search for V given Lagrangians.

        Impose the constraints
        V is sos
        V(x_equilibrium) = 0
        ∀ x s.t V(x)≤ ρ, ∃ u∈𝒰, s.t V̇(x, u) <= −κ*V

        Returns:
          prog: The optimization program.
          V: The CLF.
          rho: The sub-level set is {x | V(x) <= rho}.
        """
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x_set)
        V, _ = new_sos_polynomial(
            prog, self.x_set, V_degree, bool(np.all(self.x_equilibrium == 0))
        )
        rho = prog.NewContinuousVariables(1, "rho")[0]
        prog.AddBoundingBoxConstraint(0, np.inf, rho)
        self._add_clf_condition(prog, V, lagrangians, rho, kappa)
        return prog, V, rho

    def _add_clf_condition(
        self,
        prog: solvers.MathematicalProgram,
        V: sym.Polynomial,
        lagrangians: Union[ClfWInputLimitLagrangian, ClfWoInputLimitLagrangian],
        rho: Union[sym.Variable, float],
        kappa: float,
    ) -> sym.Polynomial:
        dVdx = V.Jacobian(self.x)
        dVdx_times_f = dVdx.dot(self.f)
        dVdx_times_g = dVdx.reshape((1, -1)) @ self.g
        if self.u_vertices is None:
            assert isinstance(lagrangians, ClfWoInputLimitLagrangian)
            sos_poly = (
                -(1 + lagrangians.dVdx_times_f) * (dVdx_times_f + kappa * V)
                - dVdx_times_g.reshape((-1,)).dot(lagrangians.dVdx_times_g)
                - lagrangians.rho_minus_V * (rho - V)
            )
        else:
            assert isinstance(lagrangians, ClfWInputLimitLagrangian)
            Vdot = (dVdx_times_f + dVdx_times_g @ self.u_vertices.T).reshape((-1,))
            sos_poly = (1 + lagrangians.V_minus_rho) * (V - rho) * sym.Polynomial(
                self.x.dot(self.x)
            ) - lagrangians.Vdot.dot(Vdot + kappa * V)
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            sos_poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
        prog.AddSosConstraint(sos_poly)
        return sos_poly


class FindClfWoInputLimitsCounterExample:
    """
    For a candidate CLF function (without input limits), find the counter-example
    (that violates the CLF condition) through nonlinear optimization.
    Namely we solve this problem:
    max ∂V/∂x*f(x)+κ*V
    s.t ∂V/∂x*g(x) = 0
        V(x) <= ρ

    This class constructs the optimization program above (but doesn't solve it).
    You can access the constructed program through self.prog.
    """

    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        V: sym.Polynomial,
        rho: float,
        kappa: float,
        state_eq_constraints: Optional[np.ndarray],
    ):
        self.x = x
        self.nx = x.size
        self.f = f
        assert f.shape == (self.nx,)
        self.g = g
        assert g.shape[0] == self.nx
        self.nu = g.shape[1]
        self.V = V
        self.dVdx = V.Jacobian(self.x)
        self.dVdx_times_f: sym.Polynomial = self.dVdx.dot(f)
        self.J_dVdx_times_f = self.dVdx_times_f.Jacobian(self.x)
        self.dVdx_times_g = (self.dVdx.reshape((1, -1)) @ self.g).reshape((-1,))
        self.J_dVdx_times_g = np.array(
            [self.dVdx_times_g[i].Jacobian(self.x) for i in range(self.nu)]
        )
        self.rho = rho
        self.kappa = kappa
        self.h = state_eq_constraints
        self.h_size = 0 if self.h is None else self.h.size
        self.J_h = (
            np.array([self.h[i].Jacobian(x) for i in range(self.h.size)])
            if self.h is not None
            else None
        )
        self.cost_poly = -(self.dVdx_times_f + self.kappa * V)
        self.J_cost_poly = -(self.J_dVdx_times_f + self.kappa * self.dVdx)

        def constraint(x_eval):
            if x_eval.dtype == object:
                x_val = pydrake.autodiffutils.ExtractValue(x_eval)
                x_grad = pydrake.autodiffutils.ExtractGradient(x_eval)
            else:
                x_val = x_eval
            env = {x[i]: x_val[i] for i in range(self.nx)}
            constraint_val = np.empty((1 + self.nu + self.h_size,))
            constraint_val[: self.nu] = np.array(
                [self.dVdx_times_g[i].Evaluate(env) for i in range(self.nu)]
            )
            constraint_val[self.nu] = self.V.Evaluate(env)
            if self.h is not None:
                constraint_val[-self.h_size :] = np.array(
                    [self.h[i].Evaluate(env) for i in range(self.h_size)]
                )
            # Now evaluate the gradient.
            if x_eval.dtype == object:
                dconstraint_dx = np.zeros((constraint_val.size, self.nx))
                dconstraint_dx[: self.nu, :] = np.array(
                    [
                        [
                            self.J_dVdx_times_g[i, j].Evaluate(env)
                            for j in range(self.nx)
                        ]
                        for i in range(self.nu)
                    ]
                )
                dconstraint_dx[self.nu, :] = np.array(
                    [self.dVdx[i].Evaluate(env) for i in range(self.nx)]
                )
                if self.h is not None:
                    dconstraint_dx[-self.h_size :, :] = np.array(
                        [
                            [self.J_h[i, j].Evaluate(env) for j in range(self.nx)]
                            for i in range(self.h_size)
                        ]
                    )
                d_constraint = dconstraint_dx @ x_grad
                return pydrake.autodiffutils.InitializeAutoDiff(
                    constraint_val, d_constraint
                )
            else:
                return constraint_val

        def cost(x_eval):
            if x_eval.dtype == object:
                x_val = pydrake.autodiffutils.ExtractValue(x_eval)
                x_grad = pydrake.autodiffutils.ExtractGradient(x_eval)
            else:
                x_val = x_eval
            env = {x[i]: x_val[i] for i in range(self.nx)}
            cost_val = self.cost_poly.Evaluate(env)

            if x_eval.dtype == object:
                cost_grad = np.array(
                    [self.J_cost_poly[i].Evaluate(env) for i in range(self.nx)]
                )
                d_cost = (cost_grad.reshape((1, -1)) @ x_grad).reshape((-1,))
                return pydrake.autodiffutils.AutoDiffXd(cost_val, d_cost)
            else:
                return cost_val

        self.prog = solvers.MathematicalProgram()
        self.prog.AddDecisionVariables(x)
        self.prog.AddCost(cost, x)
        constraint_lb = np.concatenate(
            (np.zeros((self.nu,)), np.array([-np.inf]), np.zeros((self.h_size,)))
        )
        constraint_ub = np.concatenate(
            (np.zeros((self.nu,)), np.array([self.rho]), np.zeros((self.h_size,)))
        )

        self.prog.AddConstraint(constraint, constraint_lb, constraint_ub, x)


def find_candidate_regional_lyapunov(
    x: np.ndarray,
    dynamics: np.ndarray,
    V_degree: int,
    positivity_eps: float,
    d: int,
    kappa: float,
    state_eq_constraints: np.ndarray,
    positivity_ceq_lagrangian_degrees: List[int],
    derivative_ceq_lagrangian_degrees: List[int],
    state_ineq_constraints: np.ndarray,
    positivity_cin_lagrangian_degrees: List[int],
    derivative_cin_lagrangian_degrees: List[int],
) -> Tuple[solvers.MathematicalProgram, sym.Polynomial]:
    """
    Constructs a program to find Lyapunov candidate V for a closed-loop system,
    that satisfy the following constraints within a region cin(x) <= 0

    Find V(x), p1(x), p2(x), q1(1), q2(x)
    s.t V - ε1*(xᵀx)ᵈ + p1(x) * cin(x) - p2(x) * ceq(x) is sos  (1)
       -Vdot - κ * V + q1(x) * cin(x) - q2(x) * ceq(x) is sos  (2)
       p1(x) is sos, q1(x) is sos.

    Namely SOS can verify that on the semialgebraic set
    {x | cin(x) <= 0, ceq(x) = 0}, we have V(x) >= 0 and Vdot <= -κ * V

    Args:
      dynamics: An array of polynomials of x. The closed-loop dynamics.
      state_eq_constraints: An array of polynomials of x. The left hand side of
      ceq(x) = 0
      state_ineq_constraints: An array of polynomials of x. The left hand side
      of cin(x) <= 0
    """
    # We assume that the goal state is x = 0.
    check_polynomials_pass_origin(dynamics)
    prog = solvers.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    # We know that ceq(0) = 0 (because the goal state 0 should satisfy the
    # equality constraint). Combining this with
    # V - ε1*(xᵀx)ᵈ + p1(x) * cin(x) - p2(x) * ceq(x) is sos     (1)
    # we know that the left hand side of (1) is 0 at the x=0. The left hand side
    # is p1(0) * cin(0) at x=0.
    # We know that cin(0) < 0 (assume that the goal state 0 is in the strict interior
    # of the region). p1(x) is a sos, hence p1(0) = 0. A sos polynomial with
    # constant term equal to 0 also means that all its linear terms should be 0.
    # Hence p1(x) doesn't have a linear or constant terms. Hence the only linear
    # term from the left hand side of (1) can come from V(x) and and ceq(x), and
    # the linear terms have to cancel out.
    no_linear_term_variables = find_no_linear_term_variables(
        x_set, state_eq_constraints
    )
    V = new_free_polynomial_pass_origin(
        prog, x_set, V_degree, "V", no_linear_term_variables
    )
    # Add the constraint V - ε1*(xᵀx)ᵈ + p1(x) * cin(x) - p2(x) * ceq(x) is sos
    positivity_sos_condition = V
    if positivity_eps > 0:
        positivity_sos_condition -= positivity_eps * sym.Polynomial(
            sym.pow(x.dot(x), d)
        )
    p1 = np.array(
        [
            prog.NewSosPolynomial(x_set, degree)[0]
            for degree in positivity_cin_lagrangian_degrees
        ]
    )
    positivity_sos_condition += p1.dot(state_ineq_constraints)
    p2 = np.array(
        [
            prog.NewFreePolynomial(x_set, degree)
            for degree in positivity_ceq_lagrangian_degrees
        ]
    )
    positivity_sos_condition -= p2.dot(state_eq_constraints)
    prog.AddSosConstraint(positivity_sos_condition)

    # Impose the constraint -Vdot - κ * V + q1(x) * cin(x) - q2(x) * ceq(x) is sos
    Vdot = V.Jacobian(x).dot(dynamics)
    derivative_sos_condition = -Vdot - kappa * V
    q1 = np.array(
        [
            prog.NewSosPolynomial(x_set, degree)[0]
            for degree in derivative_cin_lagrangian_degrees
        ]
    )
    derivative_sos_condition += q1.dot(state_ineq_constraints)
    q2 = np.array(
        [
            prog.NewFreePolynomial(x_set, degree)
            for degree in derivative_ceq_lagrangian_degrees
        ]
    )
    derivative_sos_condition -= q2.dot(state_eq_constraints)
    prog.AddSosConstraint(derivative_sos_condition)
    return prog, V


def save_clf(V: sym.Polynomial, x_set: sym.Variables, kappa: float, pickle_path: str):
    """
    Save the CLF to a pickle file.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}
    data["V"] = compatible_clf_cbf.utils.serialize_polynomial(V, x_set)
    data["kappa"] = kappa

    if os.path.exists(pickle_path):
        overwrite_cmd = input(
            f"File {pickle_path} already exists. Overwrite the file? Press [Y/n]:"
        )
        if overwrite_cmd in ("Y", "y"):
            save_cmd = True
        else:
            save_cmd = False
    else:
        save_cmd = True

    if save_cmd:
        with open(pickle_path, "wb") as handle:
            pickle.dump(data, handle)


def load_clf(pickle_path: str, x_set: sym.Variables) -> dict:
    ret = {}
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    ret["V"] = compatible_clf_cbf.utils.deserialize_polynomial(data["V"], x_set)
    ret["kappa"] = data["kappa"]
    return ret
