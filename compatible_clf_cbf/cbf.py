"""
Certify and search Control Barrier Function (CBF) through sum-of-squares optimization.
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

from compatible_clf_cbf.clf_cbf import (
    SafetySet,
    SafetySetLagrangians,
    SafetySetLagrangianDegrees,
)
import compatible_clf_cbf.clf_cbf as clf_cbf


@dataclass
class CbfWoInputLimitLagrangian:
    """
    The Lagrangians for proving the CBF condition for systems _without_ input limits.
    """

    # The Lagrangian λ₀(x) in (1+λ₀(x))*(∂h/∂x*f(x)+κ*h(x))
    dhdx_times_f: sym.Polynomial
    # The Lagrangian λ₁(x) in λ₁(x)*∂h/∂x*g(x)
    dhdx_times_g: np.ndarray
    # The Lagrangian λ₂(x) in λ₂(x)*(h(x)+ε)
    h_plus_eps: sym.Polynomial
    # The array of Lagrangians for state equality constraints.
    state_eq_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        dhdx_times_f_result = get_polynomial_result(
            result, self.dhdx_times_f, coefficient_tol
        )
        dhdx_times_g_result = get_polynomial_result(
            result, self.dhdx_times_g, coefficient_tol
        )
        h_plus_eps_result = get_polynomial_result(
            result, self.h_plus_eps, coefficient_tol
        )
        state_eq_constraints_result = (
            None
            if self.state_eq_constraints is None
            else get_polynomial_result(
                result, self.state_eq_constraints, coefficient_tol
            )
        )
        return CbfWoInputLimitLagrangian(
            dhdx_times_f=dhdx_times_f_result,
            dhdx_times_g=dhdx_times_g_result,
            h_plus_eps=h_plus_eps_result,
            state_eq_constraints=state_eq_constraints_result,
        )


@dataclass
class CbfWoInputLimitLagrangianDegrees:
    dhdx_times_f: int
    dhdx_times_g: List[int]
    h_plus_eps: int
    state_eq_constraints: Optional[List[int]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
    ) -> CbfWoInputLimitLagrangian:
        """
        Constructs the Lagrangians as SOS polynomials.
        """
        dhdx_times_f, _ = new_sos_polynomial(prog, x_set, self.dhdx_times_f)
        dhdx_times_g = np.array(
            [prog.NewFreePolynomial(x_set, degree) for degree in self.dhdx_times_g]
        )
        h_plus_eps, _ = new_sos_polynomial(prog, x_set, self.h_plus_eps)

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
        return CbfWoInputLimitLagrangian(
            dhdx_times_f, dhdx_times_g, h_plus_eps, state_eq_constraints
        )


@dataclass
class CbfWInputLimitLagrangian:
    """
    The Lagrangians for proving the CBF condition for systems _with_ input limits.
    """

    # The Lagrangian λ₀(x) in (1+λ₀(x))(h(x)+ε)
    h_plus_eps: sym.Polynomial
    # The Lagrangians λᵢ(x) in ∑ᵢ λᵢ(x)*(∂h/∂x*(f(x)+g(x)uᵢ)+κ*V(x))
    hdot: np.ndarray
    # The Lagrangians for state equality constraints
    state_eq_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        h_plus_eps_result: sym.Polynomial = get_polynomial_result(
            result, self.h_plus_eps, coefficient_tol
        )
        hdot_result: np.ndarray = get_polynomial_result(
            result, self.hdot, coefficient_tol
        )
        state_eq_constraints_result = (
            None
            if self.state_eq_constraints is None
            else get_polynomial_result(
                result, self.state_eq_constraints, coefficient_tol
            )
        )
        return CbfWInputLimitLagrangian(
            h_plus_eps=h_plus_eps_result,
            hdot=hdot_result,
            state_eq_constraints=state_eq_constraints_result,
        )


@dataclass
class CbfWInputLimitLagrangianDegrees:
    h_plus_eps: int
    hdot: List[int]
    state_eq_constraints: Optional[List[int]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
    ) -> CbfWInputLimitLagrangian:
        h_plus_eps, _ = new_sos_polynomial(prog, x_set, self.h_plus_eps)
        hdot = np.array(
            [new_sos_polynomial(prog, x_set, degree) for degree in self.hdot]
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
        return CbfWInputLimitLagrangian(h_plus_eps, hdot, state_eq_constraints)


class ControlBarrier:
    """
    For a control affine system
    ẋ=f(x)+g(x)u, u∈𝒰
    where the unsafe region is defined by {x | p(x) <= 0}
    we certify its CBF h(x) through SOS.

    We will need to verify that the super-level set {x | h(x)>=0} doesn't
    intersect with the unsafe region. Moreover, we prove when h(x) ≥−ε, there
    exists u∈𝒰, such that ḃ(x, u) = ∂h/∂x(f(x)+g(x)u) ≥ −κh(x).

    To prove this condition, we consider two cases:
    1) when the input u in unconstrained.
    2) when the input u is constrained within a polytope.

    If the set 𝒰 is the entire space (namely the control input is not bounded),
    then h is an CBF iff the set satisfying the following conditions
    ∂h/∂x*f(x)+κ*V(x) < 0
    ∂h/∂x*g(x)=0
    h(x)≥−ε
    is empty.

    By positivestellasatz, this means that there exists λ₀(x), λ₁(x), λ₂(x)
    satisfying
    (1+λ₀(x))*(∂h/∂x*f(x)+κ*h(x))−λ₁(x)*∂h/∂x*g(x)−λ₂(x)*(h(x)+ε) is sos.
    λ₀(x), λ₁(x), λ₂(x) are sos.

    If the set 𝒰 is a polytope with vertices u₁,..., uₙ, then h is an CBF iff
    -ε-h(x) is always non-negative on the semi-algebraic set
    {x|∂h/∂x*(f(x)+g(x)uᵢ)<= −κ*h(x), i=1,...,n}
    By positivestellasatz, we have
    -(1+λ₀(x))(h(x)+ε) + ∑ᵢ λᵢ(x)*(∂h/∂x*(f(x)+g(x)uᵢ)+κ*V(x))
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
        safety_set: SafetySet,
        u_vertices: Optional[np.ndarray] = None,
        state_eq_constraints: Optional[np.ndarray] = None
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
          safety_set: describes the exclude region and the within region.
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
        self.safety_set = safety_set
        if u_vertices is not None:
            assert u_vertices.shape[1] == self.nu
        self.u_vertices = u_vertices
        if state_eq_constraints is not None:
            check_array_of_polynomials(state_eq_constraints, self.x_set)
        self.state_eq_constraints = state_eq_constraints

    def search_lagrangians_given_cbf(
        self,
        h: sym.Polynomial,
        eps: float,
        kappa: float,
        cbf_derivative_lagrangian_degrees: Union[
            CbfWInputLimitLagrangianDegrees, CbfWoInputLimitLagrangianDegrees
        ],
        safety_set_lagrangian_degrees: SafetySetLagrangianDegrees,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
    ) -> Tuple[
        Optional[Union[CbfWInputLimitLagrangian, CbfWoInputLimitLagrangian]],
        Optional[SafetySetLagrangians],
    ]:
        """
        For a given CBF candidate, certify the CBF conditions by finding the
        Lagrangian multipliers.
        """
        if self.safety_set.exclude is not None:
            prog_exclude = solvers.MathematicalProgram()
            prog_exclude.AddIndeterminates(self.x_set)
            exclude_lagrangians = safety_set_lagrangian_degrees.exclude.to_lagrangians(
                prog_exclude, self.x_set
            )
            self._add_barrier_exclude_constraint(prog_exclude, h, exclude_lagrangians)
            result_exclude = solve_with_id(prog_exclude, solver_id, solver_options)
            if result_exclude.is_success():
                exclude_lagrangians_result = exclude_lagrangians.get_result(
                    result_exclude, lagrangian_coefficient_tol
                )
        else:
            exclude_lagrangians_result = None

        within_lagrangians_result: Optional[
            List[Optional[clf_cbf.WithinRegionLagrangians]]
        ] = None
        if self.safety_set.within is not None:
            within_lagrangians_result = [None] * (self.safety_set.within.size)
            for i in range(self.safety_set.within.size):
                prog_within = solvers.MathematicalProgram()
                prog_within.AddIndeterminates(self.x_set)
                within_lagrangians = safety_set_lagrangian_degrees.within[
                    i
                ].to_lagrangians(prog_within, self.x_set)
                self._add_barrier_within_constraint(
                    prog_within, i, h, within_lagrangians
                )
                result_within = solve_with_id(prog_within, solver_id, solver_options)
                if result_within.is_success():
                    within_lagrangians_result[i] = within_lagrangians.get_result(
                        result_within, lagrangian_coefficient_tol
                    )

        safety_set_lagrangians_result = clf_cbf.SafetySetLagrangians(
            exclude=exclude_lagrangians_result, within=within_lagrangians_result
        )

        prog_cbf_derivative = solvers.MathematicalProgram()
        prog_cbf_derivative.AddIndeterminates(self.x_set)
        cbf_derivative_lagrangians = cbf_derivative_lagrangian_degrees.to_lagrangians(
            prog_cbf_derivative, self.x_set
        )
        self._add_cbf_derivative_condition(
            prog_cbf_derivative, h, cbf_derivative_lagrangians, eps, kappa
        )
        result_cbf_derivative = solve_with_id(
            prog_cbf_derivative, solver_id, solver_options
        )

        cbf_derivative_lagrangians_result = (
            cbf_derivative_lagrangians.get_result(
                result_cbf_derivative, lagrangian_coefficient_tol
            )
            if result_cbf_derivative.is_success()
            else None
        )
        return cbf_derivative_lagrangians_result, safety_set_lagrangians_result

    def _add_barrier_within_constraint(
        self,
        prog: solvers.MathematicalProgram,
        within_index: int,
        h: sym.Polynomial,
        lagrangians: clf_cbf.WithinRegionLagrangians,
    ) -> sym.Polynomial:
        """
        Adds the constraint that the 0-super level set of the barrier function
        is in the safe region {x | pᵢ(x) <= 0}.
        −(1+ϕ₀(x))pᵢ(x) − ϕ₁(x)h(x) is sos.

        Note it doesn't add the constraints
        ϕ₀(x) is sos
        ϕ₁(x) is sos.

        Args:
          safe_region_index: pᵢ(x) = self.safety_set.within[within_index]
        """
        assert self.safety_set.within is not None
        poly = (
            -(1 + lagrangians.safe_region) * self.safety_set.within[within_index]
            - lagrangians.cbf * h
        )
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
        prog.AddSosConstraint(poly)
        return poly

    def _add_barrier_exclude_constraint(
        self,
        prog: solvers.MathematicalProgram,
        h: sym.Polynomial,
        lagrangians: clf_cbf.ExcludeRegionLagrangians,
    ) -> sym.Polynomial:
        """
        Adds the constraint that the 0-superlevel set of the barrier function
        does not intersect with the unsafe region.
        Since the i'th unsafe regions is defined as the 0-sublevel set of
        polynomials p(x), we want to certify that the set {x|p(x)≤0, h(x)≥0}
        is empty.
        The emptiness of the set can be certified by the constraint
        -(1+ϕ₀(x))h(x) +∑ⱼϕⱼ(x)pⱼ(x) is sos
        ϕ₀(x), ϕⱼ(x) are sos.

        Note that this function only adds the constraint
        -(1+ϕ₀(x))*hᵢ(x) +∑ⱼϕⱼ(x)pⱼ(x) is sos
        It doesn't add the constraint ϕ₀(x), ϕⱼ(x) are sos.

        Args:
          h: a polynomial, h is the barrier function for the
            unsafe region self.unsafe_regions[unsafe_region_index].
          lagrangians: A array of polynomials, ϕ(x) in the documentation above.
        Returns:
          poly: poly is the polynomial -(1+ϕ₀(x))hᵢ(x) + ∑ⱼϕⱼ(x)pⱼ(x)
        """
        poly = -(1 + lagrangians.cbf) * h + lagrangians.unsafe_region.dot(
            self.safety_set.exclude
        )
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
        prog.AddSosConstraint(poly)
        return poly

    def _add_cbf_derivative_condition(
        self,
        prog: solvers.MathematicalProgram,
        h: sym.Polynomial,
        lagrangians: Union[CbfWInputLimitLagrangian, CbfWoInputLimitLagrangian],
        eps: float,
        kappa: float,
    ) -> sym.Polynomial:
        """
        Add the constraint
        If u is unbounded:
        (1+λ₀(x))*(∂h/∂x*f(x)+κ*h(x))−λ₁(x)*∂h/∂x*g(x)−λ₂(x)*(h(x)+ε) is sos.
        otherwise:
        -(1+λ₀(x))(h(x)+ε) + ∑ᵢ λᵢ(x)*(∂h/∂x*(f(x)+g(x)uᵢ)+κ*V(x))
        """
        dhdx = h.Jacobian(self.x)
        dhdx_times_f = dhdx.dot(self.f)
        dhdx_times_g = dhdx.reshape((1, -1)) @ self.g
        if self.u_vertices is None:
            assert isinstance(lagrangians, CbfWoInputLimitLagrangian)
            sos_poly = (
                (1 + lagrangians.dhdx_times_f) * (dhdx_times_f + kappa * h)
                - lagrangians.dhdx_times_g.dot(dhdx_times_g.reshape((-1,)))
                - lagrangians.h_plus_eps * (h + eps)
            )
        else:
            assert isinstance(lagrangians, CbfWInputLimitLagrangian)
            hdot = (dhdx_times_f + dhdx_times_g @ self.u_vertices.T).reshape((-1,))
            sos_poly = -(1 + lagrangians.h_plus_eps) * (h + eps) + lagrangians.hdot.dot(
                hdot + kappa * h
            )
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            sos_poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
        prog.AddSosConstraint(sos_poly)
        return sos_poly


class CbfConstraint:
    """
    Add the linear constraint dhdx * f(x) + dhdx * g(x)*u >= -kappa * h(x) on u.
    """

    def __init__(
        self,
        h: sym.Polynomial,
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        kappa: float,
    ):
        dhdx = h.Jacobian(x)
        dhdx_times_f = dhdx.dot(f)
        dhdx_times_g = dhdx @ g
        self.rhs = -kappa * h - dhdx_times_f
        self.lhs_coeff = dhdx_times_g
        self.x = x

    def add_to_prog(
        self, prog: solvers.MathematicalProgram, x_val: np.ndarray, u: np.ndarray
    ) -> solvers.Binding[solvers.LinearConstraint]:
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        lhs_coeff = np.array([p.Evaluate(env) for p in self.lhs_coeff])
        rhs = self.rhs.Evaluate(env)
        constraint = prog.AddLinearConstraint(lhs_coeff, rhs, np.inf, u)
        return constraint
