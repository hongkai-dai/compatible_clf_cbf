"""
Certify and search Control Lyapunov function (CLF) through sum-of-squares optimization.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_cbf.utils import (
    check_array_of_polynomials,
    get_polynomial_result,
    new_sos_polynomial,
    solve_with_id,
)


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
            [new_sos_polynomial(prog, x_set, degree) for degree in self.Vdot]
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
    λ₀(x), λ₁(x), λ₂(x) are sos.

    If the set 𝒰 is a polytope with vertices u₁,..., uₙ, then V is an CLF iff
    (V(x)-ρ)xᵀx is always non-negative on the semi-algebraic set
    {x|∂V/∂x*(f(x)+g(x)uᵢ)≥ −κ*V(x), i=1,...,n}
    By positivestellasatz, we have
    (1+λ₀(x))(V(x)−ρ)xᵀx − ∑ᵢ λᵢ(x)*(∂V/∂x*(f(x)+g(x)uᵢ)+κ*V(x))
    λ₀(x), λᵢ(x) are sos.
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
        coefficient_tol: Optional[float] = None,
    ) -> Union[ClfWInputLimitLagrangian, ClfWoInputLimitLagrangian]:
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x_set)
        lagrangians = lagrangian_degrees.to_lagrangians(
            prog, self.x_set, self.x_equilibrium
        )
        self._add_clf_condition(prog, V, lagrangians, rho, kappa)
        result = solve_with_id(prog, solver_id, solver_options)
        assert result.is_success()
        return lagrangians.get_result(result, coefficient_tol)

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
                - dVdx_times_g.squeeze().dot(lagrangians.dVdx_times_g)
                - lagrangians.rho_minus_V * (rho - V)
            )
        else:
            assert isinstance(lagrangians, ClfWInputLimitLagrangian)
            Vdot = dVdx_times_f + dVdx_times_g @ self.u_vertices
            sos_poly = (1 + lagrangians.V_minus_rho) * (V - rho) * sym.Polynomial(
                self.x.dot(self.x)
            ) - lagrangians.Vdot.dot(Vdot + kappa * V)
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            sos_poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
        prog.AddSosConstraint(sos_poly)
        return sos_poly
