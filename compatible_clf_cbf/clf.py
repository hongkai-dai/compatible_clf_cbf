"""
Certify and search Control Lyapunov function (CLF) through sum-of-squares optimization.
"""
from dataclasses import dataclass
from typing import List, Optional, Union
from typing_extensions import Self

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_cbf.utils import (
    check_array_of_polynomials,
    get_polynomial_result,
    new_sos_polynomial,
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
    # The Lagrangian
    rho_minus_V: sym.Polynomial

    @classmethod
    def construct(
        cls,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        dVdx_times_f_degree: int,
        dVdx_times_g_degrees: List[int],
        rho_minus_V_degree: int,
        x_equilibrium: np.ndarray,
    ) -> Self:
        """
        Constructs the Lagrangians as SOS polynomials.
        """
        dVdx_times_f, _ = new_sos_polynomial(prog, x_set, dVdx_times_f_degree)
        dVdx_times_g = np.array(
            [
                new_sos_polynomial(prog, x_set, degree)[0]
                for degree in dVdx_times_g_degrees
            ]
        )
        # We know that V(x_equilibrium) = 0 and dVdx at x_equilibrium is also
        # 0. Hence by
        # -(1+λ₀(x))*(∂V/∂x*f(x)+κ*V(x))−λ₁(x)*∂V/∂x*g(x)−λ₂(x)*(ρ−V(x))
        # being sos, we know that λ₂(x_equilibrium) = 0.
        # When x_equilibrium = 0, this means that λ₂(0) = 0. Since λ₂(x) is
        # sos, we know that its monomial basis doesn't contain 1.
        if rho_minus_V_degree == 0:
            rho_minus_V = sym.Polynomial()
        else:
            if np.all(x_equilibrium == 0):
                monomial_basis = sym.MonomialBasis(
                    x_set, int(np.floor(rho_minus_V_degree / 2))
                )
                assert monomial_basis[-1].total_degree() == 0
                rho_minus_V, _ = prog.NewSosPolynomial(monomial_basis[:-1])
            else:
                rho_minus_V, _ = prog.NewSosPolynomial(x_set, rho_minus_V_degree)
        return ClfWoInputLimitLagrangian(dVdx_times_f, dVdx_times_g, rho_minus_V)

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
        return ClfWoInputLimitLagrangian(
            dVdx_times_f=dVdx_times_f_result,
            dVdx_times_g=dVdx_times_g_result,
            rho_minus_V=rho_minus_V_result,
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
        return ClfWInputLimitLagrangian(
            V_minus_rho=V_minus_rho_result, Vdot=Vdot_result
        )


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
            prog.AddSosConstraint(sos_poly)
        else:
            assert isinstance(lagrangians, ClfWInputLimitLagrangian)
            Vdot = dVdx_times_f + dVdx_times_g @ self.u_vertices
            sos_poly = (1 + lagrangians.V_minus_rho) * (V - rho) * sym.Polynomial(
                self.x.dot(self.x)
            ) - lagrangians.Vdot.dot(Vdot + kappa * V)
            prog.AddSosConstraint(sos_poly)
        return sos_poly
