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
    UnsafeRegionLagrangians,
    UnsafeRegionLagrangianDegrees,
)


@dataclass
class CbfWoInputLimitLagrangian:
    """
    The Lagrangians for proving the CBF condition for systems _without_ input limits.
    """

    # The Lagrangian λ₀(x) in (1+λ₀(x))*(∂b/∂x*f(x)+κ*b(x))
    dbdx_times_f: sym.Polynomial
    # The Lagrangian λ₁(x) in λ₁(x)*∂b/∂x*g(x)
    dbdx_times_g: np.ndarray
    # The Lagrangian λ₂(x) in λ₂(x)*(b(x)+ε)
    b_plus_eps: sym.Polynomial
    # The array of Lagrangians for state equality constraints.
    state_eq_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        dbdx_times_f_result = get_polynomial_result(
            result, self.dbdx_times_f, coefficient_tol
        )
        dbdx_times_g_result = get_polynomial_result(
            result, self.dbdx_times_g, coefficient_tol
        )
        b_plus_eps_result = get_polynomial_result(
            result, self.b_plus_eps, coefficient_tol
        )
        state_eq_constraints_result = (
            None
            if self.state_eq_constraints is None
            else get_polynomial_result(
                result, self.state_eq_constraints, coefficient_tol
            )
        )
        return CbfWoInputLimitLagrangian(
            dbdx_times_f=dbdx_times_f_result,
            dbdx_times_g=dbdx_times_g_result,
            b_plus_eps=b_plus_eps_result,
            state_eq_constraints=state_eq_constraints_result,
        )


@dataclass
class CbfWoInputLimitLagrangianDegrees:
    dbdx_times_f: int
    dbdx_times_g: List[int]
    b_plus_eps: int
    state_eq_constraints: Optional[List[int]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
    ) -> CbfWoInputLimitLagrangian:
        """
        Constructs the Lagrangians as SOS polynomials.
        """
        dbdx_times_f, _ = new_sos_polynomial(prog, x_set, self.dbdx_times_f)
        dbdx_times_g = np.array(
            [prog.NewFreePolynomial(x_set, degree) for degree in self.dbdx_times_g]
        )
        b_plus_eps, _ = new_sos_polynomial(prog, x_set, self.b_plus_eps)

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
            dbdx_times_f, dbdx_times_g, b_plus_eps, state_eq_constraints
        )


@dataclass
class CbfWInputLimitLagrangian:
    """
    The Lagrangians for proving the CBF condition for systems _with_ input limits.
    """

    # The Lagrangian λ₀(x) in (1+λ₀(x))(b(x)+ε)
    b_plus_eps: sym.Polynomial
    # The Lagrangians λᵢ(x) in ∑ᵢ λᵢ(x)*(∂b/∂x*(f(x)+g(x)uᵢ)+κ*V(x))
    bdot: np.ndarray
    # The Lagrangians for state equality constraints
    state_eq_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        b_plus_eps_result: sym.Polynomial = get_polynomial_result(
            result, self.b_plus_eps, coefficient_tol
        )
        bdot_result: np.ndarray = get_polynomial_result(
            result, self.bdot, coefficient_tol
        )
        state_eq_constraints_result = (
            None
            if self.state_eq_constraints is None
            else get_polynomial_result(
                result, self.state_eq_constraints, coefficient_tol
            )
        )
        return CbfWInputLimitLagrangian(
            b_plus_eps=b_plus_eps_result,
            bdot=bdot_result,
            state_eq_constraints=state_eq_constraints_result,
        )


@dataclass
class CbfWInputLimitLagrangianDegrees:
    b_plus_eps: int
    bdot: List[int]
    state_eq_constraints: Optional[List[int]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
    ) -> CbfWInputLimitLagrangian:
        b_plus_eps, _ = new_sos_polynomial(prog, x_set, self.b_plus_eps)
        bdot = np.array(
            [new_sos_polynomial(prog, x_set, degree) for degree in self.bdot]
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
        return CbfWInputLimitLagrangian(b_plus_eps, bdot, state_eq_constraints)


class ControlBarrier:
    """
    For a control affine system
    ẋ=f(x)+g(x)u, u∈𝒰
    where the unsafe region is defined by {x | p(x) <= 0}
    we certify its CBF b(x) through SOS.

    We will need to verify that the super-level set {x | b(x)>=0} doesn't
    intersect with the unsafe region. Moreover, we prove when b(x) ≥−ε, there
    exists u∈𝒰, such that ḃ(x, u) = ∂b/∂x(f(x)+g(x)u) ≥ −κb(x).

    To prove this condition, we consider two cases:
    1) when the input u in unconstrained.
    2) when the input u is constrained within a polytope.

    If the set 𝒰 is the entire space (namely the control input is not bounded),
    then b is an CBF iff the set satisfying the following conditions
    ∂b/∂x*f(x)+κ*V(x) < 0
    ∂b/∂x*g(x)=0
    b(x)≥−ε
    is empty.

    By positivestellasatz, this means that there exists λ₀(x), λ₁(x), λ₂(x)
    satisfying
    (1+λ₀(x))*(∂b/∂x*f(x)+κ*b(x))−λ₁(x)*∂b/∂x*g(x)−λ₂(x)*(b(x)+ε) is sos.
    λ₀(x), λ₁(x), λ₂(x) are sos.

    If the set 𝒰 is a polytope with vertices u₁,..., uₙ, then b is an CBF iff
    -ε-b(x) is always non-negative on the semi-algebraic set
    {x|∂b/∂x*(f(x)+g(x)uᵢ)<= −κ*b(x), i=1,...,n}
    By positivestellasatz, we have
    -(1+λ₀(x))(b(x)+ε) + ∑ᵢ λᵢ(x)*(∂b/∂x*(f(x)+g(x)uᵢ)+κ*V(x))
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
        unsafe_region: np.ndarray,
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
          unsafe_region: An array of polynomials. The unsafe region is
            {x|unsafe_region(x) <= 0}.
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
        check_array_of_polynomials(unsafe_region, self.x_set)
        self.unsafe_region = unsafe_region
        if u_vertices is not None:
            assert u_vertices.shape[1] == self.nu
        self.u_vertices = u_vertices
        if state_eq_constraints is not None:
            check_array_of_polynomials(state_eq_constraints, self.x_set)
        self.state_eq_constraints = state_eq_constraints

    def search_lagrangians_given_cbf(
        self,
        b: sym.Polynomial,
        eps: float,
        kappa: float,
        cbf_derivative_lagrangian_degrees: Union[
            CbfWInputLimitLagrangianDegrees, CbfWoInputLimitLagrangianDegrees
        ],
        unsafe_region_lagrangian_degrees: UnsafeRegionLagrangianDegrees,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
    ) -> Tuple[
        Optional[Union[CbfWInputLimitLagrangian, CbfWoInputLimitLagrangian]],
        Optional[UnsafeRegionLagrangians],
    ]:
        """
        For a given CBF candidate, certify the CBF conditions by finding the
        Lagrangian multipliers.
        """
        prog_unsafe = solvers.MathematicalProgram()
        prog_unsafe.AddIndeterminates(self.x_set)
        unsafe_lagrangians = unsafe_region_lagrangian_degrees.to_lagrangians(
            prog_unsafe, self.x_set
        )
        self._add_barrier_safe_constraint(prog_unsafe, b, unsafe_lagrangians)
        result_unsafe = solve_with_id(prog_unsafe, solver_id, solver_options)
        if result_unsafe.is_success():
            unsafe_lagrangians_result = unsafe_lagrangians.get_result(
                result_unsafe, lagrangian_coefficient_tol
            )
        else:
            unsafe_lagrangians_result = None

        prog_cbf_derivative = solvers.MathematicalProgram()
        prog_cbf_derivative.AddIndeterminates(self.x_set)
        cbf_derivative_lagrangians = cbf_derivative_lagrangian_degrees.to_lagrangians(
            prog_cbf_derivative, self.x_set
        )
        self._add_cbf_derivative_condition(
            prog_cbf_derivative, b, cbf_derivative_lagrangians, eps, kappa
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
        return cbf_derivative_lagrangians_result, unsafe_lagrangians_result

    def _add_barrier_safe_constraint(
        self,
        prog: solvers.MathematicalProgram,
        b: sym.Polynomial,
        lagrangians: UnsafeRegionLagrangians,
    ) -> sym.Polynomial:
        """
        Adds the constraint that the 0-superlevel set of the barrier function
        does not intersect with the unsafe region.
        Since the i'th unsafe regions is defined as the 0-sublevel set of
        polynomials p(x), we want to certify that the set {x|p(x)≤0, b(x)≥0}
        is empty.
        The emptiness of the set can be certified by the constraint
        -(1+ϕ₀(x))b(x) +∑ⱼϕⱼ(x)pⱼ(x) is sos
        ϕ₀(x), ϕⱼ(x) are sos.

        Note that this function only adds the constraint
        -(1+ϕ₀(x))*bᵢ(x) +∑ⱼϕⱼ(x)pⱼ(x) is sos
        It doesn't add the constraint ϕ₀(x), ϕⱼ(x) are sos.

        Args:
          b: a polynomial, b is the barrier function for the
            unsafe region self.unsafe_regions[unsafe_region_index].
          lagrangians: A array of polynomials, ϕ(x) in the documentation above.
        Returns:
          poly: poly is the polynomial -(1+ϕ₀(x))bᵢ(x) + ∑ⱼϕⱼ(x)pⱼ(x)
        """
        poly = -(1 + lagrangians.cbf) * b + lagrangians.unsafe_region.dot(
            self.unsafe_region
        )
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
        prog.AddSosConstraint(poly)
        return poly

    def _add_cbf_derivative_condition(
        self,
        prog: solvers.MathematicalProgram,
        b: sym.Polynomial,
        lagrangians: Union[CbfWInputLimitLagrangian, CbfWoInputLimitLagrangian],
        eps: float,
        kappa: float,
    ) -> sym.Polynomial:
        """
        Add the constraint
        If u is unbounded:
        (1+λ₀(x))*(∂b/∂x*f(x)+κ*b(x))−λ₁(x)*∂b/∂x*g(x)−λ₂(x)*(b(x)+ε) is sos.
        otherwise:
        -(1+λ₀(x))(b(x)+ε) + ∑ᵢ λᵢ(x)*(∂b/∂x*(f(x)+g(x)uᵢ)+κ*V(x))
        """
        dbdx = b.Jacobian(self.x)
        dbdx_times_f = dbdx.dot(self.f)
        dbdx_times_g = dbdx.reshape((1, -1)) @ self.g
        if self.u_vertices is None:
            assert isinstance(lagrangians, CbfWoInputLimitLagrangian)
            sos_poly = (
                (1 + lagrangians.dbdx_times_f) * (dbdx_times_f + kappa * b)
                - lagrangians.dbdx_times_g.dot(dbdx_times_g.squeeze())
                - lagrangians.b_plus_eps * (b + eps)
            )
        else:
            assert isinstance(lagrangians, CbfWInputLimitLagrangian)
            bdot = dbdx_times_f + dbdx_times_g @ self.u_vertices
            sos_poly = -(1 + lagrangians.b_plus_eps) * (b + eps) + lagrangians.bdot.dot(
                bdot + kappa * b
            )
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            sos_poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
        prog.AddSosConstraint(sos_poly)
        return sos_poly
