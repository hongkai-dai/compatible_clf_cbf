"""
Verifying and synthesizing compatible CLF/CBF.

Reference:
Verification and Synthesis of Compatible Control Lyapunov and Control Barrier
Functions
Hongkai Dai*, Chuanrui Jiang*, Hongchao Zhang and Andrew Clark.
IEEE Conference on Decision and Control (CDC), 2024
"""

from dataclasses import dataclass
import os
import os.path
import pickle
from typing import List, Optional, Tuple
from typing_extensions import Self

import numpy as np

import pydrake.symbolic as sym
import pydrake.solvers as solvers

from compatible_clf_cbf.utils import (
    BinarySearchOptions,
    ContainmentLagrangianDegree,
    check_array_of_polynomials,
    get_polynomial_result,
    new_sos_polynomial,
    solve_with_id,
)
import compatible_clf_cbf.utils
import compatible_clf_cbf.ellipsoid_utils as ellipsoid_utils


@dataclass
class CompatibleLagrangians:
    """
    The Lagrangians for proving the compatibility condition, namely set (1) or (2)
    defined in CompatibleClfCbf class documentation is empty.
    """

    # An array of symbolic polynomials. The Lagrangian multiplies with Λ(x)ᵀy if
    # use_y_squared = False, or Λ(x)ᵀy² if use_y_squared = True.
    # Each entry in this Lagrangian multiplier is a free polynomial.
    # Size is (nu,)
    lambda_y: np.ndarray
    # The Lagrangian polynomial multiplies with ξ(x)ᵀy if use_y_squared = False,
    # or ξ(x)ᵀy² if use_y_squared = True. This multiplier is a free polynomial.
    xi_y: sym.Polynomial
    # The Lagrangian polynomial multiplies with y if use_y_squared = False.
    # This multiplier is an array of SOS polynomials.
    y: Optional[np.ndarray]
    # The Lagrangian polynomial multiplies with ρ − V when with_clf = True, and
    # we search for an CLF with a region-of-attraction {x | V(x) <= ρ}.
    # Should be a SOS polynomial.
    rho_minus_V: Optional[sym.Polynomial]
    # The Lagrangian polynomials multiplies with h(x)+ε. Should be an array of SOS
    # polynomials.
    h_plus_eps: Optional[np.ndarray]
    # The free Lagrangian polynomials multiplying the state equality
    # constraints.
    state_eq_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        """
        Gets the result of the Lagrangians.
        """
        lambda_y_result = get_polynomial_result(result, self.lambda_y, coefficient_tol)
        xi_y_result = get_polynomial_result(result, self.xi_y, coefficient_tol)
        y_result = (
            get_polynomial_result(result, self.y, coefficient_tol)
            if self.y is not None
            else None
        )
        rho_minus_V_result = (
            get_polynomial_result(result, self.rho_minus_V, coefficient_tol)
            if self.rho_minus_V is not None
            else None
        )
        h_plus_eps_result = (
            get_polynomial_result(result, self.h_plus_eps, coefficient_tol)
            if self.h_plus_eps is not None
            else None
        )
        state_eq_constraints_result = (
            get_polynomial_result(result, self.state_eq_constraints, coefficient_tol)
            if self.state_eq_constraints is not None
            else None
        )
        return CompatibleLagrangians(
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            y=y_result,
            rho_minus_V=rho_minus_V_result,
            h_plus_eps=h_plus_eps_result,
            state_eq_constraints=state_eq_constraints_result,
        )


@dataclass
class CompatibleLagrangianDegrees:
    """
    The degree of the Lagrangian multipliers in CompatibleLagrangians.
    """

    @dataclass
    class Degree:
        """
        The degree of each Lagrangian polynomial in indeterminates x and y. For
        example, if we have a polynomial x₀²x₁y₂ + 3x₀y₁y₂³, its degree in x is
        3 (from x₀²x₁), and its degree in y is 4 (from y₁y₂³)
        """

        x: int
        y: int

        def construct_polynomial(
            self,
            prog: solvers.MathematicalProgram,
            x: sym.Variables,
            y: sym.Variables,
            is_sos: bool,
            sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        ) -> sym.Polynomial:
            """
            Args:
              is_sos: whether the constructed polynomial is sos or not.
            """
            if is_sos:
                basis = sym.MonomialBasis(
                    {x: int(np.floor(self.x / 2)), y: int(np.floor(self.y / 2))}
                )
                poly, _ = prog.NewSosPolynomial(basis, type=sos_type)
            else:
                basis = sym.MonomialBasis({x: self.x, y: self.y})
                coeffs = prog.NewContinuousVariables(basis.size)
                poly = sym.Polynomial({basis[i]: coeffs[i] for i in range(basis.size)})
            return poly

    lambda_y: List[Degree]
    xi_y: Degree
    y: Optional[List[Degree]]
    rho_minus_V: Optional[Degree]
    h_plus_eps: Optional[List[Degree]]
    state_eq_constraints: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        y: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lambda_y_lagrangian: Optional[np.ndarray] = None,
        xi_y_lagrangian: Optional[sym.Polynomial] = None,
        y_lagrangian: Optional[np.ndarray] = None,
        rho_minus_V_lagrangian: Optional[sym.Polynomial] = None,
        h_plus_eps_lagrangian: Optional[np.ndarray] = None,
        state_eq_constraints_lagrangian: Optional[np.ndarray] = None,
    ) -> CompatibleLagrangians:
        if lambda_y_lagrangian is None:
            lambda_y = np.array(
                [
                    lambda_y_i.construct_polynomial(prog, x, y, is_sos=False)
                    for lambda_y_i in self.lambda_y
                ]
            )
        else:
            lambda_y = lambda_y_lagrangian
        if xi_y_lagrangian is None:
            xi_y = self.xi_y.construct_polynomial(prog, x, y, is_sos=False)
        else:
            xi_y = xi_y_lagrangian
        if y_lagrangian is None:
            y_lagrangian = (
                None
                if self.y is None
                else np.array(
                    [
                        y_i.construct_polynomial(
                            prog, x, y, is_sos=True, sos_type=sos_type
                        )
                        for y_i in self.y
                    ]
                )
            )
        if rho_minus_V_lagrangian is None:
            rho_minus_V = (
                None
                if self.rho_minus_V is None
                else self.rho_minus_V.construct_polynomial(
                    prog, x, y, is_sos=True, sos_type=sos_type
                )
            )
        else:
            rho_minus_V = rho_minus_V_lagrangian
        if h_plus_eps_lagrangian is None:
            h_plus_eps = (
                None
                if self.h_plus_eps is None
                else np.array(
                    [
                        h_plus_eps_i.construct_polynomial(
                            prog, x, y, is_sos=True, sos_type=sos_type
                        )
                        for h_plus_eps_i in self.h_plus_eps
                    ]
                )
            )
        else:
            h_plus_eps = h_plus_eps_lagrangian
        if state_eq_constraints_lagrangian is None:
            state_eq_constraints = (
                None
                if self.state_eq_constraints is None
                else np.array(
                    [
                        state_eq_constraints_i.construct_polynomial(
                            prog, x, y, is_sos=False
                        )
                        for state_eq_constraints_i in self.state_eq_constraints
                    ]
                )
            )
        else:
            state_eq_constraints = state_eq_constraints_lagrangian
        return CompatibleLagrangians(
            lambda_y=lambda_y,
            xi_y=xi_y,
            y=y_lagrangian,
            rho_minus_V=rho_minus_V,
            h_plus_eps=h_plus_eps,
            state_eq_constraints=state_eq_constraints,
        )


@dataclass
class WithinRegionLagrangians:
    """
    The Lagrangians for certifying that the 0-super level set of a CBF is a
    subset of the safe region.

    for a CBF h(x), to prove that the 0-super level set {x | h(x) >= 0} is a
    subset of the safe set {x | p(x) <= 0}, we can impose the constraint
    −(1+ϕ₀(x))p(x) − ϕ₁(x)h(x) is sos.
    ϕ₀(x) is sos
    ϕ₁(x) is sos.
    """

    cbf: sym.Polynomial
    safe_region: sym.Polynomial
    state_eq_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        return WithinRegionLagrangians(
            cbf=get_polynomial_result(result, self.cbf, coefficient_tol),
            safe_region=get_polynomial_result(
                result, self.safe_region, coefficient_tol
            ),
            state_eq_constraints=(
                None
                if self.state_eq_constraints is None
                else get_polynomial_result(
                    result, self.state_eq_constraints, coefficient_tol
                )
            ),
        )


@dataclass
class WithinRegionLagrangianDegrees:
    cbf: int
    safe_region: int
    state_eq_constraints: Optional[List[int]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        cbf_lagrangian: Optional[sym.Polynomial] = None,
    ) -> WithinRegionLagrangians:
        if cbf_lagrangian is None:
            cbf, _ = new_sos_polynomial(prog, x_set, self.cbf)
        else:
            cbf = cbf_lagrangian

        safe_region = new_sos_polynomial(prog, x_set, self.safe_region)[0]

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
        return WithinRegionLagrangians(
            cbf=cbf,
            safe_region=safe_region,
            state_eq_constraints=state_eq_constraints,
        )


@dataclass
class ExcludeRegionLagrangians:
    """
    The Lagrangians for certifying that the 0-super level set of a CBF doesn't
    intersect with an unsafe region.

    For a CBF function hᵢ(x), to prove that the 0-super level set
    {x |hᵢ(x) >= 0} doesn't intersect with an unsafe set
    {x | pⱼ(x) <= 0 for all j}, we impose the condition:

    -(1+ϕᵢ,₀(x))*hᵢ(x) +∑ⱼϕᵢ,ⱼ(x)pⱼ(x) is sos
    ϕᵢ,₀(x), ϕᵢ,ⱼ(x) are sos.
    """

    # The Lagrangian that multiplies with CBF function.
    # ϕᵢ,₀(x) in the documentation above.
    cbf: sym.Polynomial
    # An array of sym.Polynomial. The Lagrangians that multiply the unsafe region
    # polynomials. ϕᵢ,ⱼ(x) in the documentation above.
    unsafe_region: np.ndarray
    # The free Lagrangian that multiplies with the state equality constraints
    # (such as sin²θ+cos²θ=1)
    state_eq_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        return ExcludeRegionLagrangians(
            cbf=get_polynomial_result(result, self.cbf, coefficient_tol),
            unsafe_region=get_polynomial_result(
                result, self.unsafe_region, coefficient_tol
            ),
            state_eq_constraints=(
                None
                if self.state_eq_constraints is None
                else get_polynomial_result(
                    result, self.state_eq_constraints, coefficient_tol
                )
            ),
        )


@dataclass
class ExcludeRegionLagrangianDegrees:
    cbf: int
    unsafe_region: List[int]
    state_eq_constraints: Optional[List[int]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        cbf_lagrangian: Optional[sym.Polynomial] = None,
    ) -> ExcludeRegionLagrangians:
        if cbf_lagrangian is None:
            cbf, _ = new_sos_polynomial(prog, x_set, self.cbf)
        else:
            cbf = cbf_lagrangian

        unsafe_region = np.array(
            [
                new_sos_polynomial(prog, x_set, degree)[0]
                for degree in self.unsafe_region
            ]
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
        return ExcludeRegionLagrangians(
            cbf=cbf,
            unsafe_region=unsafe_region,
            state_eq_constraints=state_eq_constraints,
        )


@dataclass
class ExcludeSet:
    """
    An exclude set is described as {x | lᵢ(x)<=0 for all i}. Namely it is a
    semi-algebraic set. This is the "unsafe set" that the system state should not be in.
    """

    l: np.ndarray


@dataclass
class WithinSet:
    """
    A within set is described as {x | lᵢ(x)<=0 for all i}. Namely it is a
    semi-algebraic set. This is the "safe set" that the system state has to be within.
    """

    l: np.ndarray


@dataclass
class SafetySetLagrangians:
    # exclude[i] is the the i'th exclude set.
    exclude: List[ExcludeRegionLagrangians]
    # within[i] is for the within_set.l[i]
    within: List[WithinRegionLagrangians]

    def contains_none(self) -> bool:
        """
        Returns true if self.exclude or self.within contains None
        """
        return not (all(self.exclude) and all(self.within))

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        exclude = [a.get_result(result, coefficient_tol) for a in self.exclude]
        within = [a.get_result(result, coefficient_tol) for a in self.within]
        return SafetySetLagrangians(exclude=exclude, within=within)

    def get_cbf_lagrangians(
        self,
    ) -> Tuple[List[sym.Polynomial], List[sym.Polynomial]]:
        exclude_cbf = [exclude.cbf for exclude in self.exclude]
        within_cbf = [within.cbf for within in self.within]
        return exclude_cbf, within_cbf


@dataclass
class SafetySetLagrangianDegrees:
    exclude: List[ExcludeRegionLagrangianDegrees]
    within: List[WithinRegionLagrangianDegrees]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        cbf_lagrangian: Optional[
            Tuple[List[sym.Polynomial], List[sym.Polynomial]]
        ] = None,
    ) -> SafetySetLagrangians:
        exclude = [
            self.exclude[i].to_lagrangians(
                prog, x_set, (None if cbf_lagrangian is None else cbf_lagrangian[0][i])
            )
            for i in range(len(self.exclude))
        ]
        within = [
            self.within[i].to_lagrangians(
                prog, x_set, None if cbf_lagrangian is None else cbf_lagrangian[1][i]
            )
            for i in range(len(self.within))
        ]
        return SafetySetLagrangians(exclude=exclude, within=within)


@dataclass
class InnerEllipsoidOptions:
    """
    This option is used to encourage the compatible region to cover an inscribed
    ellipsoid.
    """

    # A state that should be contained in the inscribed ellipsoid
    x_inner: np.ndarray
    # when we search for the ellipsoid, we put a trust region constraint. This
    # is the squared radius of that trust region.
    ellipsoid_trust_region: float
    # We enlarge the inner ellipsoid through a sequence of SDPs. This is the max
    # number of iterations in that sequence.
    find_inner_ellipsoid_max_iter: int

    def __init__(
        self,
        x_inner: np.ndarray,
        ellipsoid_trust_region: float = 100.0,
        find_inner_ellipsoid_max_iter: int = 3,
    ):
        self.x_inner = x_inner
        self.ellipsoid_trust_region = ellipsoid_trust_region
        self.find_inner_ellipsoid_max_iter = find_inner_ellipsoid_max_iter


@dataclass
class CompatibleStatesOptions:
    """
    This option is used to encourage the compatible region to include certain
    candidate states. Namely h(x_candidate) >= 0 and V(x_candidate) <= 1.
    """

    candidate_compatible_states: np.ndarray
    # To avoid arbitrarily scaling the CBF, we need to impose the
    # constraint that
    # h_anchor_bounds[i][0] <= h[i](anchor_states) <= h_anchor_bounds[i][1]
    anchor_states: Optional[np.ndarray]
    h_anchor_bounds: Optional[List[Tuple[np.ndarray, np.ndarray]]]

    # To encourage the compatible region to cover the candidate states, we add
    # this cost
    # weight_V * ReLU(V(x_candidates) - (1-V_margin) )
    #    + weight_h[i] * ReLU(-h[i](x_candidates) + h_margins[i])
    weight_V: Optional[float]
    weight_h: np.ndarray
    # If not None, then we penalize the violation of V <= 1 - V_margin
    V_margin: Optional[float] = None
    # If not None, then we penalize the violation of h[i] >= h_margins[i]
    h_margins: Optional[np.ndarray] = None

    def add_cost(
        self,
        prog: solvers.MathematicalProgram,
        x: np.ndarray,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
    ) -> Tuple[solvers.Binding[solvers.LinearCost], Optional[np.ndarray], np.ndarray]:
        """
        Adds the cost
        weight_V * ReLU(V(x_candidates) - 1 + V_margin)
           + weight_h[i] * ReLU(-h[i](x_candidates) + h_margins[i])
        """
        assert h.shape == self.weight_h.shape
        num_candidates = self.candidate_compatible_states.shape[0]
        if V is not None:
            # Add the slack variable representing ReLU(V(x_candidates)-1 + V_margin)
            V_relu = prog.NewContinuousVariables(num_candidates, "V_relu")
            prog.AddBoundingBoxConstraint(0, np.inf, V_relu)
            # Now evaluate V(x_candidates) as A_v * V_decision_vars + b_v
            A_v, V_decision_vars, b_v = V.EvaluateWithAffineCoefficients(
                x, self.candidate_compatible_states.T
            )
            # Now impose the constraint V_relu >= V(x_candidates) - 1 + V_margin as
            # V_relu - A_v * V_decision_vars >= b_v -1 + V_margin
            prog.AddLinearConstraint(
                np.concatenate((-A_v, np.eye(num_candidates)), axis=1),
                b_v - 1 + (0 if self.V_margin is None else self.V_margin),
                np.full_like(b_v, np.inf),
                np.concatenate((V_decision_vars, V_relu)),
            )
        else:
            V_relu = None
        # Add the slack variable h_relu[i] representing ReLU(-h[i](x_candidates))
        h_relu = prog.NewContinuousVariables(h.shape[0], num_candidates, "h_relu")
        prog.AddBoundingBoxConstraint(0, np.inf, h_relu.reshape((-1,)))
        for i in range(h.shape[0]):
            A_h, h_decision_vars, b_h = h[i].EvaluateWithAffineCoefficients(
                x, self.candidate_compatible_states.T
            )
            # Now impose the constraint
            # h_relu[i] >= -h[i](x_candidates) + h_margins[i] as
            # A_h * h_decision_vars + h_relu[i] >= - b_h + h_margins[i]
            prog.AddLinearConstraint(
                np.concatenate((A_h, np.eye(num_candidates)), axis=1),
                -b_h + (0 if self.h_margins is None else self.h_margins[i]),
                np.full_like(b_h, np.inf),
                np.concatenate((h_decision_vars, h_relu[i])),
            )

        cost_coeff = (self.weight_h.reshape((-1, 1)) * np.ones_like(h_relu)).reshape(
            (-1,)
        )
        cost_vars = h_relu.reshape((-1,))
        if V is not None:
            assert self.weight_V is not None
            cost_coeff = np.concatenate(
                (cost_coeff, self.weight_V * np.ones(num_candidates))
            )
            assert V_relu is not None
            cost_vars = np.concatenate((cost_vars, V_relu))
        cost = prog.AddLinearCost(cost_coeff, 0.0, cost_vars)
        return cost, V_relu, h_relu

    def add_constraint(
        self, prog: solvers.MathematicalProgram, x: np.ndarray, h: np.ndarray
    ) -> Optional[List[solvers.Binding[solvers.LinearConstraint]]]:
        """
        Add the constraint
        h_anchor_bounds[i][0] <= h[i](anchor_states) <= h_anchor_bounds[i][1]
        """
        if self.h_anchor_bounds is not None:
            assert h.shape == (len(self.h_anchor_bounds),)
            assert self.anchor_states is not None
            constraints: List[solvers.Binding[solvers.LinearConstraint]] = [None] * len(
                self.h_anchor_bounds
            )
            for i in range(len(self.h_anchor_bounds)):
                assert (
                    self.h_anchor_bounds[i][0].size
                    == self.h_anchor_bounds[i][1].size
                    == self.anchor_states.shape[0]
                )
                # Evaluate h[i](anchor_states) as A_h * decision_vars_h + b_h
                A_h, decision_vars_h, b_h = h[i].EvaluateWithAffineCoefficients(
                    x, self.anchor_states.T
                )
                # Adds the constraint
                constraints[i] = prog.AddLinearConstraint(
                    A_h,
                    self.h_anchor_bounds[i][0] - b_h,
                    self.h_anchor_bounds[i][1] - b_h,
                    decision_vars_h,
                )
            return constraints
        return None


class CompatibleClfCbf:
    """
    Certify and synthesize compatible Control Lyapunov Function (CLF) and
    Control Barrier Functions (CBFs).

    For a continuous-time control-affine system
    ẋ = f(x)+g(x)u, u∈𝒰
    A CLF V(x) and a CBF h(x) is compatible if and only if
    ∃ u∈𝒰,      ∂h/∂x*f(x) + ∂h/∂x*g(x)*u ≥ −κ_h*h(x)
            and ∂V/∂x*f(x) + ∂V/∂x*g(x)*u ≤ −κ_V*V(x)
    For simplicity, let's first consider that u is un-constrained, namely 𝒰 is
    the entire space.
    By Farkas lemma, this is equivalent to the following set being empty

    {(x, y) | [y(0)]ᵀ*[-∂h/∂x*g(x)] = 0, [y(0)]ᵀ*[ ∂h/∂x*f(x)+κ_h*h(x)] = -1, y>=0}        (1)
              [y(1)]  [ ∂V/∂x*g(x)]      [y(1)]  [-∂V/∂x*f(x)-κ_V*V(x)]

    We can then use Positivstellensatz to certify the emptiness of this set.

    The same math applies to multiple CBFs, or when u is constrained within a
    polyhedron.

    If u is constrained within a polytope {u | Au * u <= bu}, we know that there exists
    u in the polytope satisfying the CLF and CBF condition, iff the following set is
    empty

    {(x, y) | yᵀ * [-∂h/∂x*g(x)] = 0, yᵀ * [ ∂h/∂x*f(x)+κ_h*h(x)] = -1 }                   (2)
                   [ ∂V/∂x*g(x)]           [-∂V/∂x*f(x)-κ_V*V(x)]
                   [         Au]           [                 bu ]
    Namely we increase the dimensionality of y and append the equality condition in (1)
    with Au and bu.
    """  # noqa E501

    def __init__(
        self,
        *,
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        exclude_sets: List[ExcludeSet],
        within_set: Optional[WithinSet],
        Au: Optional[np.ndarray] = None,
        bu: Optional[np.ndarray] = None,
        num_cbf: int = 1,
        with_clf: bool = True,
        use_y_squared: bool = True,
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
          exclude_sets: List[ExcludeSet]
            A list of "unsafe sets", namely the union of these ExcludeSet is the unsafe region.
          within_set: Optional[WithinSet]
            If not None, then the state has to be within this `within_set`.
          Au: Optional[np.ndarray]
            The set of admissible control is Au * u <= bu.
            The shape is (Any, nu)
          bu: Optional[np.ndarray]
            The set of admissible control is Au * u <= bu.
            The shape is (Any,)
          num_cbf: int
            The number of CBF functions. We require these CBF functions to be
            compatible in the intersection of their 0-superlevel-set.
          with_clf: bool
            Whether to certify or search for CLF. If set to False, then we will
            certify or search multiple compatible CBFs without CLF.
          use_y_squared: bool
            For that empty set in the class documentation, we could replace
            y>=0 condition with using y². This will potentially reduce the
            number of Lagrangian multipliers in the p-satz, but increase the
            total degree of the polynomials. Set use_y_squared=True if we use
            y², and we certify the set

            {(x, y) | [y(0)²]ᵀ*[-∂h/∂x*g(x)] = 0, [y(0)²]ᵀ*[ ∂h/∂x*f(x)+κ_h*h(x)] = -1}       (2)
                      [y(1)²]  [ ∂V/∂x*g(x)]      [y(1)²]  [-∂V/∂x*f(x)-κ_V*V(x)]
            is empty.
          state_eq_constraints: An array of polynomials. Some dynamical systems
            have equality constraints on its states. For example, when the
            state include sinθ and cosθ (so that the dynamics is a polynomial
            function of state), we need to impose the equality constraint
            sin²θ+cos²θ=1 on the state. state_eq_constraints[i] = 0 is an
            equality constraint on the state.

          If both Au and bu are None, it means that we don't have input limits.
          They have to be both None or both not None.
        """  # noqa E501
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
        self.exclude_sets = exclude_sets
        self.within_set = within_set
        if Au is not None:
            assert Au.shape[1] == self.nu
            assert bu is not None
            assert bu.shape == (Au.shape[0],)
        self.Au = Au
        self.bu = bu
        self.with_clf = with_clf
        self.use_y_squared = use_y_squared
        self.num_cbf = num_cbf
        y_size = (
            self.num_cbf
            + (1 if self.with_clf else 0)
            + (self.Au.shape[0] if self.Au is not None else 0)
        )
        self.y: np.ndarray = sym.MakeVectorContinuousVariable(y_size, "y")
        self.y_set: sym.Variables = sym.Variables(self.y)
        self.xy_set: sym.Variables = sym.Variables(np.concatenate((self.x, self.y)))
        # y_poly[i] is just the polynomial y[i]. I wrote it in this more complicated
        # form to save some computation.
        self.y_poly = np.array(
            [sym.Polynomial(sym.Monomial(self.y[i], 1)) for i in range(y_size)]
        )
        # y_squared_poly[i] is just the polynomial y[i]**2.
        self.y_squared_poly = np.array(
            [sym.Polynomial(sym.Monomial(self.y[i], 2)) for i in range(y_size)]
        )
        self.state_eq_constraints = state_eq_constraints
        if self.state_eq_constraints is not None:
            check_array_of_polynomials(self.state_eq_constraints, self.x_set)

    def certify_cbf_safety_set(
        self,
        cbf: sym.Polynomial,
        lagrangian_degrees: SafetySetLagrangianDegrees,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
    ) -> Optional[SafetySetLagrangians]:
        """
        Certify the 0-super level set of the barrier function cbf(x) is within
        the safety set.
        """
        assert len(self.exclude_sets) == len(lagrangian_degrees.exclude)

        exclude_lagrangians = [
            self.certify_cbf_exclude(
                i,
                cbf,
                lagrangian_degrees.exclude[i],
                solver_id,
                solver_options,
                lagrangian_coefficient_tol,
            )
            for i in range(len(self.exclude_sets))
        ]
        if not all(exclude_lagrangians):
            return None
        if self.within_set is None:
            assert len(lagrangian_degrees.within) == 0
            within_lagrangians = []
        else:
            within_lagrangians = [
                self.certify_cbf_within(
                    i,
                    cbf,
                    lagrangian_degrees.within[i],
                    solver_id,
                    solver_options,
                    lagrangian_coefficient_tol,
                )
                for i in range(len(self.within_set.l))
            ]
            if not all(within_lagrangians):
                return None

        return SafetySetLagrangians(
            exclude=exclude_lagrangians, within=within_lagrangians
        )

    def certify_cbf_within(
        self,
        within_index: int,
        cbf: sym.Polynomial,
        lagrangian_degrees: WithinRegionLagrangianDegrees,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
    ) -> Optional[WithinRegionLagrangians]:
        """
        Certify the 0-super level set of barrier function h(x) is within the
        region {x|self.within_set.l[within_index](x) <= 0}
        """
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x_set)
        lagrangians = lagrangian_degrees.to_lagrangians(prog, self.x_set)
        self._add_barrier_within_constraint(prog, within_index, cbf, lagrangians)
        result = solve_with_id(prog, solver_id, solver_options)
        lagrangians_result = (
            lagrangians.get_result(result, lagrangian_coefficient_tol)
            if result.is_success()
            else None
        )
        return lagrangians_result

    def certify_cbf_exclude(
        self,
        exclude_set_index: int,
        cbf: sym.Polynomial,
        lagrangian_degrees: ExcludeRegionLagrangianDegrees,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
    ) -> Optional[ExcludeRegionLagrangians]:
        """
        Certifies that the 0-superlevel set {x | hᵢ(x) >= 0} does not intersect
        with the unsafe region {x | self.exclude_sets[exclude_set_index].l(x) <= 0}

        If we denote the unsafe region as {x | p(x) <= 0}, then we impose the constraint

        We impose the constraint
        -(1+ϕᵢ,₀(x))*hᵢ(x) +∑ⱼϕᵢ,ⱼ(x)pⱼ(x) is sos
        ϕᵢ,₀(x), ϕᵢ,ⱼ(x) are sos.

        Args:
          exclude_set_index: We certify the CBF for the region
            {x | self.exclude_sets[exclude_set_index].l(x) <= 0}
          cbf: hᵢ(x) in the documentation above.
        """
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x_set)
        lagrangians = lagrangian_degrees.to_lagrangians(prog, self.x_set)
        self._add_barrier_exclude_constraint(prog, exclude_set_index, cbf, lagrangians)
        result = solve_with_id(prog, solver_id, solver_options)
        lagrangians_result = (
            lagrangians.get_result(result, lagrangian_coefficient_tol)
            if result.is_success()
            else None
        )
        return lagrangians_result

    def construct_search_compatible_lagrangians(
        self,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
        kappa_V: Optional[float],
        kappa_h: np.ndarray,
        lagrangian_degrees: CompatibleLagrangianDegrees,
        barrier_eps: Optional[np.ndarray],
        local_clf: bool = True,
        lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Tuple[solvers.MathematicalProgram, CompatibleLagrangians]:
        """
        Given CLF candidate V and CBF candidate h, construct the optimization
        program to certify that they are compatible within the region
        {x | V(x) <= 1} ∩ {x | h(x) >= -eps}.

        Args:
          V: The CLF candidate. If empty, then we will certify that the multiple
            barrier functions are compatible.
          h: The CBF candidates.
          kappa_V: The exponential decay rate for CLF. Namely we want V̇ ≤ −κ_V*V
          kappa_h: The exponential rate for CBF, namely we want ḃ ≥ −κ_h*h
          lagrangian_degrees: The degrees for the Lagrangian polynomials.
          barrier_eps: The certified safe region is {x | h(x) >= -eps}
          coefficient_tol: In the Lagrangian polynomials, we will remove the
            coefficients no larger than this tolerance.
          local_clf: Whether the CLF is valid in a local region or globally.
        Returns:
          result: The result for solving the optimization program.
          lagrangian_result: The result of the Lagrangian polynomials.
        """
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)
        lagrangians = lagrangian_degrees.to_lagrangians(
            prog, self.x_set, self.y_set, sos_type=lagrangian_sos_type
        )
        self._add_compatibility(
            prog=prog,
            V=V,
            h=h,
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            lagrangians=lagrangians,
            barrier_eps=barrier_eps,
            local_clf=local_clf,
            sos_type=compatible_sos_type,
        )
        return (prog, lagrangians)

    def search_lagrangians_given_clf_cbf(
        self,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
        kappa_V: Optional[float],
        kappa_h: np.ndarray,
        barrier_eps: np.ndarray,
        compatible_lagrangian_degrees: CompatibleLagrangianDegrees,
        safety_set_lagrangian_degrees: List[SafetySetLagrangianDegrees],
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
        lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Tuple[
        Optional[CompatibleLagrangians],
        List[Optional[SafetySetLagrangians]],
    ]:
        (
            prog_compatible,
            compatible_lagrangians,
        ) = self.construct_search_compatible_lagrangians(
            V,
            h,
            kappa_V,
            kappa_h,
            compatible_lagrangian_degrees,
            barrier_eps,
            local_clf=True,
            lagrangian_sos_type=lagrangian_sos_type,
            compatible_sos_type=compatible_sos_type,
        )
        result_compatible = solve_with_id(prog_compatible, solver_id, solver_options)
        compatible_lagrangians_result = (
            compatible_lagrangians.get_result(
                result_compatible, lagrangian_coefficient_tol
            )
            if result_compatible.is_success()
            else None
        )

        safety_set_lagrangians_result: List[Optional[SafetySetLagrangians]] = [
            None
        ] * self.num_cbf
        for i in range(self.num_cbf):
            safety_set_lagrangians_result[i] = self.certify_cbf_safety_set(
                h[i],
                safety_set_lagrangian_degrees[i],
                solver_id,
                solver_options,
                lagrangian_coefficient_tol,
            )
        return compatible_lagrangians_result, safety_set_lagrangians_result

    def search_clf_cbf_given_lagrangian(
        self,
        compatible_lagrangians: CompatibleLagrangians,
        compatible_lagrangian_degrees: CompatibleLagrangianDegrees,
        safety_sets_lagrangians: List[SafetySetLagrangians],
        safety_sets_lagrangian_degrees: List[SafetySetLagrangianDegrees],
        clf_degree: Optional[int],
        cbf_degrees: List[int],
        x_equilibrium: Optional[np.ndarray],
        kappa_V: Optional[float],
        kappa_h: np.ndarray,
        barrier_eps: np.ndarray,
        *,
        ellipsoid_inner: Optional[ellipsoid_utils.Ellipsoid] = None,
        compatible_states_options: Optional[CompatibleStatesOptions] = None,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        backoff_rel_scale: Optional[float] = None,
        backoff_abs_scale: Optional[float] = None,
        compatible_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,  # noqa
    ) -> Tuple[
        Optional[sym.Polynomial],
        Optional[np.ndarray],
        solvers.MathematicalProgramResult,
    ]:
        """
        Given the Lagrangian multipliers and an inner ellipsoid, find the clf
        and cbf, such that the compatible region contains that inner ellipsoid.

        Returns: (V, h, result)
          V: The CLF result.
          h: The CBF result.
          result: The result of the optimization program.
        """
        prog, V, h = self._construct_search_clf_cbf_program(
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            clf_degree,
            cbf_degrees,
            x_equilibrium,
            kappa_V,
            kappa_h,
            barrier_eps,
            compatible_sos_type=compatible_sos_type,
            compatible_lagrangian_sos_type=compatible_lagrangian_sos_type,
        )

        if ellipsoid_inner is not None:
            self._add_ellipsoid_in_compatible_region_constraint(
                prog, V, h, ellipsoid_inner.S, ellipsoid_inner.b, ellipsoid_inner.c
            )
        elif compatible_states_options is not None:
            self._add_compatible_states_options(prog, V, h, compatible_states_options)

        result = solve_with_id(
            prog, solver_id, solver_options, backoff_rel_scale, backoff_abs_scale
        )
        if result.is_success():
            V_sol = None if V is None else result.GetSolution(V)
            h_sol = np.array([result.GetSolution(h_i) for h_i in h])
        else:
            V_sol = None
            h_sol = None
        return V_sol, h_sol, result

    def binary_search_clf_cbf(
        self,
        compatible_lagrangians: CompatibleLagrangians,
        compatible_lagrangian_degrees: CompatibleLagrangianDegrees,
        safety_sets_lagrangians: List[SafetySetLagrangians],
        safety_sets_lagrangian_degrees: List[SafetySetLagrangianDegrees],
        clf_degree: Optional[int],
        cbf_degrees: List[int],
        x_equilibrium: Optional[np.ndarray],
        kappa_V: Optional[float],
        kappa_h: np.ndarray,
        barrier_eps: np.ndarray,
        ellipsoid_inner: ellipsoid_utils.Ellipsoid,
        scale_options: BinarySearchOptions,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
    ) -> Tuple[Optional[sym.Polynomial], np.ndarray]:
        """
        Given the Lagrangian multipliers, find the compatible CLF and CBFs,
        with the goal to enlarge the compatible region.

        We measure the size of the compatible region through binary searching
        the inner ellipsoid. We scale the inner ellipsoid about its center,
        and binary search on the scaling factor.

        Args:
          scale_options: The options to do binary search on the scale of the einner
          ellipsoid.

        Return: (V, h)
        """
        assert isinstance(scale_options, BinarySearchOptions)

        def search(
            scale,
        ) -> Tuple[
            Optional[sym.Polynomial],
            Optional[np.ndarray],
            solvers.MathematicalProgramResult,
        ]:
            c_new = ellipsoid_utils.scale_ellipsoid(
                ellipsoid_inner.S, ellipsoid_inner.b, ellipsoid_inner.c, scale
            )
            V, h, result = self.search_clf_cbf_given_lagrangian(
                compatible_lagrangians,
                compatible_lagrangian_degrees,
                safety_sets_lagrangians,
                safety_sets_lagrangian_degrees,
                clf_degree,
                cbf_degrees,
                x_equilibrium,
                kappa_V,
                kappa_h,
                barrier_eps,
                ellipsoid_inner=ellipsoid_utils.Ellipsoid(
                    ellipsoid_inner.S, ellipsoid_inner.b, c_new
                ),
                compatible_states_options=None,
                solver_id=solver_id,
                solver_options=solver_options,
            )
            return V, h, result

        scale_options.check()

        scale_min = scale_options.min
        scale_max = scale_options.max
        scale_tol = scale_options.tol

        V, h, result = search(scale_max)
        if result.is_success():
            print(f"binary_search_clf_cbf: scale={scale_max} is feasible.")
            assert h is not None
            return V, h

        V_success, h_success, result = search(scale_min)
        assert (
            result.is_success()
        ), f"binary_search_clf_cbf: scale_min={scale_min} is not feasible."
        assert h_success is not None

        while scale_max - scale_min > scale_tol:
            scale = (scale_max + scale_min) / 2
            V, h, result = search(scale)
            if result.is_success():
                print(f"binary_search_clf_cbf: scale={scale} is feasible.")
                scale_min = scale
                V_success = V
                assert h is not None
                h_success = h
            else:
                print(f"binary_search_clf_cbf: scale={scale} is not feasible.")
                scale_max = scale

        return V_success, h_success

    def in_compatible_region(
        self,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
        x_samples: np.ndarray,
    ) -> np.ndarray:
        """
        Returns if x_samples[i] is in the compatible region
        {x | V(x) <= 1, h(x) >= 0}.

        Return:
        in_compatible_flag: in_compatible_flag[i] is True iff x_samples[i] is
          in the compatible region.
        """
        in_h = np.all(
            np.concatenate(
                [
                    (h_i.EvaluateIndeterminates(self.x, x_samples.T) >= 0).reshape(
                        (-1, 1)
                    )
                    for h_i in h
                ],
                axis=1,
            ),
            axis=1,
        )
        if V is not None:
            in_V = V.EvaluateIndeterminates(self.x, x_samples.T) <= 1
            return np.logical_and(in_h, in_V)
        else:
            return in_h

    def bilinear_alternation(
        self,
        V_init: Optional[sym.Polynomial],
        h_init: np.ndarray,
        compatible_lagrangian_degrees: CompatibleLagrangianDegrees,
        safety_sets_lagrangian_degrees: List[SafetySetLagrangianDegrees],
        kappa_V: Optional[float],
        kappa_h: np.ndarray,
        barrier_eps: np.ndarray,
        x_equilibrium: np.ndarray,
        clf_degree: Optional[int],
        cbf_degrees: List[int],
        max_iter: int,
        *,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
        inner_ellipsoid_options: Optional[InnerEllipsoidOptions] = None,
        binary_search_scale_options: Optional[BinarySearchOptions] = None,
        compatible_states_options: Optional[CompatibleStatesOptions] = None,
        backoff_scale: Optional[compatible_clf_cbf.utils.BackoffScale] = None,
        lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Tuple[Optional[sym.Polynomial], np.ndarray]:
        """
        Synthesize the compatible CLF and CBF through bilinear alternation. We
        alternate between
        1. Fixing the CLF/CBF, searching for Lagrangians.
        2. Fixing Lagrangians, searching for CLF/CBF.

        Our goal is to find the compatible CLF and CBFs with the largest compatible
        region. We cannot measure the volume of the compatible region directly, so we
        use one of the following heuristics to grow the compatible region:
        1. Grow the inscribed ellipsoid within the compatible region.
        2. Expand the compatible region to cover some candidate states.

        I strongly recommend using heuristics 2 (covering candidate states)
        instead of 1 (grow the inscribed ellipsoid).

        Args:
          max_iter: The maximal number of bilinear alternation iterations.
          lagrangian_coefficient_tol: We remove the coefficients whose absolute
            value is smaller than this tolerance in the Lagrangian polynomials.
            Use None to preserve all coefficients.
        """

        # One and only one of inner_ellipsoid_options and compatible_states_options is
        # None.
        assert (
            inner_ellipsoid_options is not None and compatible_states_options is None
        ) or (inner_ellipsoid_options is None and compatible_states_options is not None)
        if inner_ellipsoid_options is not None:
            assert binary_search_scale_options is not None
            raise Warning(
                "inner_ellipsoid_options isn't None. This will grow the "
                "inscribed ellipsoid. I strongly recommend to use the other "
                "heuristics to cover candidate states, by setting "
                "compatible_states_options."
            )
        assert isinstance(binary_search_scale_options, Optional[BinarySearchOptions])
        assert isinstance(compatible_states_options, Optional[CompatibleStatesOptions])

        iteration = 0
        clf = V_init
        assert len(h_init) == self.num_cbf
        cbf = h_init

        compatible_lagrangians = None
        safety_sets_lagrangians: List[Optional[SafetySetLagrangians]] = [
            None
        ] * self.num_cbf

        def evaluate_compatible_states(clf_fun, cbf_funs, x_val):
            if clf_fun is not None:
                V_candidates = clf_fun.EvaluateIndeterminates(self.x, x_val.T)
                print(f"V(candidate_compatible_states)={V_candidates}")
            h_candidates = [
                h_i.EvaluateIndeterminates(
                    self.x,
                    x_val.T,
                )
                for h_i in cbf_funs
            ]
            for i, h_candidates_val in enumerate(h_candidates):
                print(f"h[{i}](candidate_compatible_states)={h_candidates_val}")

        for iteration in range(max_iter):
            print(f"iteration {iteration}")
            if compatible_states_options is not None:
                evaluate_compatible_states(
                    clf, cbf, compatible_states_options.candidate_compatible_states
                )
            # Search for the Lagrangians.
            (
                compatible_lagrangians,
                safety_sets_lagrangians,
            ) = self.search_lagrangians_given_clf_cbf(
                clf,
                cbf,
                kappa_V,
                kappa_h,
                barrier_eps,
                compatible_lagrangian_degrees,
                safety_sets_lagrangian_degrees,
                solver_id,
                solver_options,
                lagrangian_coefficient_tol,
                lagrangian_sos_type=lagrangian_sos_type,
                compatible_sos_type=compatible_sos_type,
            )
            assert compatible_lagrangians is not None
            assert all(safety_sets_lagrangians)

            if inner_ellipsoid_options is not None:
                # We use the heuristics to grow the inner ellipsoid.
                assert compatible_states_options is None
                # Search for the inner ellipsoid.
                V_contain_ellipsoid_lagrangian_degree = (
                    self._get_V_contain_ellipsoid_lagrangian_degree(clf)
                )
                h_contain_ellipsoid_lagrangian_degree = (
                    self._get_h_contain_ellipsoid_lagrangian_degrees(cbf)
                )
                (
                    S_ellipsoid_inner,
                    b_ellipsoid_inner,
                    c_ellipsoid_inner,
                ) = self._find_max_inner_ellipsoid(
                    clf,
                    cbf,
                    V_contain_ellipsoid_lagrangian_degree,
                    h_contain_ellipsoid_lagrangian_degree,
                    inner_ellipsoid_options.x_inner,
                    solver_id=solver_id,
                    solver_options=solver_options,
                    max_iter=inner_ellipsoid_options.find_inner_ellipsoid_max_iter,
                    trust_region=inner_ellipsoid_options.ellipsoid_trust_region,
                )

                assert binary_search_scale_options is not None
                clf, cbf = self.binary_search_clf_cbf(
                    compatible_lagrangians,
                    compatible_lagrangian_degrees,
                    safety_sets_lagrangians,
                    clf_degree,
                    cbf_degrees,
                    x_equilibrium,
                    kappa_V,
                    kappa_h,
                    barrier_eps,
                    ellipsoid_utils.Ellipsoid(
                        S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner
                    ),
                    binary_search_scale_options,
                    solver_id,
                    solver_options,
                )
            else:
                # We use the heuristics to cover some candidate states with the
                # compatible region.
                assert compatible_states_options is not None
                clf, cbf, result = self.search_clf_cbf_given_lagrangian(
                    compatible_lagrangians,
                    compatible_lagrangian_degrees,
                    safety_sets_lagrangians,
                    safety_sets_lagrangian_degrees,
                    clf_degree,
                    cbf_degrees,
                    x_equilibrium,
                    kappa_V,
                    kappa_h,
                    barrier_eps,
                    ellipsoid_inner=None,
                    compatible_states_options=compatible_states_options,
                    solver_id=solver_id,
                    solver_options=solver_options,
                    backoff_rel_scale=(
                        None if backoff_scale is None else backoff_scale.rel
                    ),
                    backoff_abs_scale=(
                        None if backoff_scale is None else backoff_scale.abs
                    ),
                    compatible_sos_type=compatible_sos_type,
                    compatible_lagrangian_sos_type=lagrangian_sos_type,
                )
                assert cbf is not None
        if compatible_states_options is not None:
            evaluate_compatible_states(
                clf, cbf, compatible_states_options.candidate_compatible_states
            )
        return clf, cbf

    def check_compatible_at_state(
        self,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
        x_val: np.ndarray,
        kappa_V: Optional[float],
        kappa_h: np.ndarray,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
    ) -> Tuple[bool, solvers.MathematicalProgramResult]:
        """
        Check if at a given state the CLF and CBFs are compatible, namely there
        exists a common u such that
        Vdot(x, u) <= -kappa_V * V
        hdot(x, u) >= -kappa_h * h
        """
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(self.nu, "u")
        if self.Au is not None:
            assert self.bu is not None
            prog.AddLinearConstraint(
                self.Au, np.full_like(self.bu, -np.inf), self.bu, u
            )
        assert x_val.shape == (self.nx,)
        env = {self.x[i]: x_val[i] for i in range(self.nx)}
        f_val = np.array([f_i.Evaluate(env) for f_i in self.f])
        g_val = np.array(
            [
                [self.g[i, j].Evaluate(env) for j in range(self.nu)]
                for i in range(self.nx)
            ]
        )
        if V is not None:
            assert kappa_V is not None
            V_val = V.Evaluate(env)
            dVdx = V.Jacobian(self.x)
            dVdx_val = np.array([dVdx[i].Evaluate(env) for i in range(self.nx)])
            dVdx_times_f_val = dVdx_val.dot(f_val)
            dVdx_times_g_val = dVdx_val @ g_val
            prog.AddLinearConstraint(
                dVdx_times_f_val + dVdx_times_g_val @ u <= -kappa_V * V_val
            )
        for h_i in h:
            h_val = h_i.Evaluate(env)
            dhdx = h_i.Jacobian(self.x)
            dhdx_val = np.array([dhdx[i].Evaluate(env) for i in range(self.nx)])
            dhdx_times_f_val = dhdx_val.dot(f_val)
            dhdx_times_g_val = dhdx_val @ g_val
            prog.AddLinearConstraint(
                dhdx_times_f_val + dhdx_times_g_val @ u >= -kappa_h * h_val
            )
        if self.Au is not None and self.bu is not None:
            prog.AddLinearConstraint(
                self.Au, np.full_like(self.bu, -np.inf), self.bu, u
            )
        result = solve_with_id(prog, solver_id, solver_options)
        return result.is_success(), result

    def _calc_xi_Lambda(
        self,
        *,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
        kappa_V: Optional[float],
        kappa_h: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute
        Λ(x) = [-∂h/∂x*g(x)]
               [ ∂V/∂x*g(x)]
               [        Au ]
        ξ(x) = [ ∂h/∂x*f(x)+κ_h*h(x)]
               [-∂V/∂x*f(x)-κ_V*V(x)]
               [                 bu ]

        Args:
          V: The CLF function. If with_clf is False, then V is None.
          h: An array of CBFs. h[i] is the CBF for the i'th unsafe region.
          kappa_V: κ_V in the documentation above.
          kappa_h: κ_h in the documentation above. kappa_h[i] is the kappa for h[i].
        Returns:
          (xi, lambda_mat) ξ(x) and Λ(x) in the documentation above.
        """
        if self.with_clf:
            assert V is not None
            assert isinstance(V, sym.Polynomial)
            dVdx = V.Jacobian(self.x)
            xi_rows = self.num_cbf + 1
        else:
            assert V is None
            dVdx = None
            xi_rows = self.num_cbf
            assert h.size > 1, "You should use multiple CBF when with_clf is False."
        if self.Au is not None:
            xi_rows += self.Au.shape[0]
        assert h.shape == (self.num_cbf,)
        assert kappa_h.shape == h.shape
        dhdx = np.concatenate(
            [h[i].Jacobian(self.x).reshape((1, -1)) for i in range(h.size)], axis=0
        )
        lambda_mat = np.empty((xi_rows, self.nu), dtype=object)
        lambda_mat[: self.num_cbf] = -dhdx @ self.g
        xi = np.empty((xi_rows,), dtype=object)
        xi[: self.num_cbf] = dhdx @ self.f + kappa_h * h

        if self.with_clf:
            assert V is not None
            assert dVdx is not None
            lambda_mat[self.num_cbf] = dVdx @ self.g
            xi[self.num_cbf] = -dVdx.dot(self.f) - kappa_V * V
        if self.Au is not None:
            lambda_mat[-self.Au.shape[0] :] = self.Au
            xi[-self.Au.shape[0] :] = self.bu

        return (xi, lambda_mat)

    def _add_compatibility(
        self,
        *,
        prog: solvers.MathematicalProgram,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
        kappa_V: Optional[float],
        kappa_h: np.ndarray,
        lagrangians: CompatibleLagrangians,
        barrier_eps: Optional[np.ndarray],
        local_clf: bool,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> sym.Polynomial:
        """
        Add the p-satz condition that certifies the following set is empty
        if use_y_squared = False:
        {(x, y) | [y(0)]ᵀ*[-∂h/∂x*g(x)] = [0], [y(0)]ᵀ*[ ∂h/∂x*f(x)+κ_h*h(x)] = -1, y>=0, V(x)≤ρ, h(x)≥−ε}         (1)
                  [y(1)]  [ ∂V/∂x*g(x)]   [0]  [y(1)]  [-∂V/∂x*f(x)-κ_V*V(x)]
        if use_y_squared = True:
        {(x, y) | [y(0)²]ᵀ*[-∂h/∂x*g(x)] = [0], [y(0)²]ᵀ*[ ∂h/∂x*f(x)+κ_h*h(x)] = -1, V(x)≤ρ, h(x)≥−ε}              (2)
                  [y(1)²]  [ ∂V/∂x*g(x)]   [0]  [y(1)²]  [-∂V/∂x*f(x)-κ_V*V(x)]
        namely inside the set {x | V(x)≤ρ, h(x)≥−ε}, the CLF and CBF are compatible.

        Let's denote
        Λ(x) = [-∂h/∂x*g(x)]
               [ ∂V/∂x*g(x)]
        ξ(x) = [ ∂h/∂x*f(x)+κ_h*h(x)]
               [-∂V/∂x*f(x)-κ_V*V(x)]
        To certify the emptiness of the set in (1), we can use the sufficient condition
        -1 - s₀(x, y)ᵀ Λ(x)ᵀy - s₁(x, y)(ξ(x)ᵀy+1) - s₂(x, y)ᵀy - s₃(x, y)(1 − V) - s₄(x, y)ᵀ(h(x)+ε) is sos          (3)
        s₂(x, y), s₃(x, y), s₄(x, y) are all sos.

        To certify the emptiness of the set in (2), we can use the sufficient condition
        -1 - s₀(x, y)ᵀ Λ(x)ᵀy² - s₁(x, y)(ξ(x)ᵀy²+1) - s₃(x, y)(1 − V) - s₄(x, y)ᵀ(h(x)+ε) is sos                     (4)
        s₃(x, y), s₄(x, y) are all sos.

        Note that we do NOT add the constraint
        s₂(x, y), s₃(x, y), s₄(x, y) are all sos.
        in this function. The user should add this constraint separately.

        Returns:
          poly: The polynomial on the left hand side of equation (3) or (4).
        """  # noqa: E501
        xi, lambda_mat = self._calc_xi_Lambda(
            V=V, h=h, kappa_V=kappa_V, kappa_h=kappa_h
        )
        # This is just polynomial 1.
        poly_one = sym.Polynomial(sym.Monomial())

        poly = -poly_one
        # Compute s₀(x, y)ᵀ Λ(x)ᵀy
        if self.use_y_squared:
            lambda_y = lambda_mat.T @ self.y_squared_poly
        else:
            lambda_y = lambda_mat.T @ self.y_poly
        poly -= lagrangians.lambda_y.dot(lambda_y)

        # Compute s₁(x, y)(ξ(x)ᵀy+1)
        if self.use_y_squared:
            xi_y = xi.dot(self.y_squared_poly) + poly_one
        else:
            xi_y = xi.dot(self.y_poly) + poly_one
        poly -= lagrangians.xi_y * xi_y

        # Compute s₂(x, y)ᵀy
        if not self.use_y_squared:
            assert lagrangians.y is not None
            poly -= lagrangians.y.dot(self.y_poly)

        # Compute s₃(x, y)(1 − V)
        if self.with_clf and local_clf:
            assert V is not None
            assert lagrangians.rho_minus_V is not None
            poly -= lagrangians.rho_minus_V * (poly_one - V)

        # Compute s₄(x, y)ᵀ(h(x)+ε)
        if barrier_eps is not None:
            assert np.all(barrier_eps >= 0)
            assert lagrangians.h_plus_eps is not None
            poly -= lagrangians.h_plus_eps.dot(barrier_eps + h)

        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)

        prog.AddSosConstraint(poly, sos_type)
        return poly

    def _add_barrier_within_constraint(
        self,
        prog: solvers.MathematicalProgram,
        within_index: int,
        h: sym.Polynomial,
        lagrangians: WithinRegionLagrangians,
    ) -> sym.Polynomial:
        """
        Adds the constraint that the 0-super level set of the barrier function
        is in the safe region {x | pᵢ(x) <= 0}.
        −(1+ϕ₀(x))pᵢ(x) − ϕ₁(x)h(x) is sos.

        Note it doesn't add the constraints
        ϕ₀(x) is sos
        ϕ₁(x) is sos.

        Args:
          within_index: pᵢ(x) = self.within_set.l[within_index]
        """
        assert self.within_set is not None
        poly = (
            -(1 + lagrangians.safe_region) * self.within_set.l[within_index]
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
        exclude_set_index: int,
        h: sym.Polynomial,
        lagrangians: ExcludeRegionLagrangians,
    ) -> sym.Polynomial:
        """
        Adds the constraint that the 0-superlevel set of the barrier function
        does not intersect with the exclude region.
        Since the i'th unsafe regions is defined as the 0-sublevel set of
        polynomials p(x), we want to certify that the set {x|p(x)≤0, hᵢ(x)≥0}
        is empty.
        The emptiness of the set can be certified by the constraint
        -(1+ϕᵢ,₀(x))hᵢ(x) +∑ⱼϕᵢ,ⱼ(x)pⱼ(x) is sos
        ϕᵢ,₀(x), ϕᵢ,ⱼ(x) are sos.

        Note that this function only adds the constraint
        -(1+ϕᵢ,₀(x))*hᵢ(x) +∑ⱼϕᵢ,ⱼ(x)pⱼ(x) is sos
        It doesn't add the constraint ϕᵢ,₀(x), ϕᵢ,ⱼ(x) are sos.

        Args:
          exclude_set_index: We certify that the 0-superlevel set of the
            barrier function doesn't intersect with the exclude region
            self.exclude_sets[exclude_set_index].l
          h: a polynomial, h is the barrier function for the
            exclude region self.exclude_sets[exclude_set_index].
          lagrangians: A array of polynomials, ϕᵢ(x) in the documentation above.
        Returns:
          poly: poly is the polynomial -(1+ϕᵢ,₀(x))hᵢ(x) + ∑ⱼϕᵢ,ⱼ(x)pⱼ(x)
        """
        assert lagrangians.unsafe_region.size == len(
            self.exclude_sets[exclude_set_index].l
        )
        poly = -(1 + lagrangians.cbf) * h + lagrangians.unsafe_region.dot(
            self.exclude_sets[exclude_set_index].l
        )
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
        prog.AddSosConstraint(poly)
        return poly

    def _construct_search_clf_cbf_program(
        self,
        compatible_lagrangians: CompatibleLagrangians,
        compatible_lagrangian_degrees: CompatibleLagrangianDegrees,
        safety_sets_lagrangians: List[SafetySetLagrangians],
        safety_sets_lagrangian_degrees: List[SafetySetLagrangianDegrees],
        clf_degree: Optional[int],
        cbf_degrees: List[int],
        x_equilibrium: Optional[np.ndarray],
        kappa_V: Optional[float],
        kappa_h: np.ndarray,
        barrier_eps: np.ndarray,
        local_clf: bool = True,
        compatible_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,  # noqa
    ) -> Tuple[
        solvers.MathematicalProgram,
        Optional[sym.Polynomial],
        np.ndarray,
    ]:
        """
        Construct a program to search for compatible CLF/CBFs given the Lagrangians.
        Notice that we have not imposed the cost to the program yet.

        Args:
          compatible_lagrangians: The Lagrangian polynomials. Result from
            solving construct_search_compatible_lagrangians().
          safety_sets_lagrangians: The Lagrangians certifying that the 0-super
            level set of the i'th CBF is in the safety set.
          clf_degree: if not None, the total degree of CLF.
          cbf_degrees: cbf_degrees[i] is the total degree of the i'th CBF.
          x_equilibrium: if not None, the equilibrium state.
        """
        assert len(safety_sets_lagrangians) == self.num_cbf
        assert len(cbf_degrees) == self.num_cbf
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)

        if clf_degree is not None:
            assert x_equilibrium is not None
            V = new_sos_polynomial(
                prog, self.x_set, clf_degree, zero_at_origin=np.all(x_equilibrium == 0)
            )[0]
            if np.any(x_equilibrium != 0):
                # Add the constraint V(x*) = 0
                (
                    V_x_equilibrium_coeff,
                    V_x_equilibrium_var,
                    V_x_equilibrium_constant,
                ) = V.EvaluateWithAffineCoefficients(
                    self.x, x_equilibrium.reshape((-1, 1))
                )
                prog.AddLinearEqualityConstraint(
                    V_x_equilibrium_coeff.reshape((1, -1)),
                    -V_x_equilibrium_constant[0],
                    V_x_equilibrium_var,
                )
        else:
            V = None

        # Add CBF.
        h = np.array(
            [
                prog.NewFreePolynomial(self.x_set, cbf_degree)
                for cbf_degree in cbf_degrees
            ]
        )
        # We can search for the Lagrangians for the safety set as well, since
        # the safety set is fixed.
        safety_sets_lagrangians_new: List[SafetySetLagrangians] = [None] * self.num_cbf
        for i in range(self.num_cbf):
            cbf_lagrangian = safety_sets_lagrangians[i].get_cbf_lagrangians()
            safety_sets_lagrangians_new[i] = safety_sets_lagrangian_degrees[
                i
            ].to_lagrangians(prog, self.x_set, cbf_lagrangian)
            assert len(safety_sets_lagrangians_new[i].exclude) == len(self.exclude_sets)
            for exclude_set_index in range(len(self.exclude_sets)):
                self._add_barrier_exclude_constraint(
                    prog,
                    exclude_set_index,
                    h[i],
                    safety_sets_lagrangians_new[i].exclude[exclude_set_index],
                )
            if self.within_set is not None:
                for j in range(self.within_set.l.size):
                    self._add_barrier_within_constraint(
                        prog, j, h[i], safety_sets_lagrangians_new[i].within[j]
                    )

        # We can search for some compatible Lagrangians as well, including the
        # Lagrangians for y >= 0 and the state equality constraints, as y>= 0
        # and the state equality constraints don't depend on V or h.
        compatible_lagrangians_new = compatible_lagrangian_degrees.to_lagrangians(
            prog,
            self.x_set,
            self.y_set,
            sos_type=compatible_lagrangian_sos_type,
            lambda_y_lagrangian=compatible_lagrangians.lambda_y,
            xi_y_lagrangian=compatible_lagrangians.xi_y,
            rho_minus_V_lagrangian=compatible_lagrangians.rho_minus_V,
            h_plus_eps_lagrangian=compatible_lagrangians.h_plus_eps,
        )

        self._add_compatibility(
            prog=prog,
            V=V,
            h=h,
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            lagrangians=compatible_lagrangians_new,
            barrier_eps=barrier_eps,
            local_clf=local_clf,
            sos_type=compatible_sos_type,
        )

        return (prog, V, h)

    def _find_max_inner_ellipsoid(
        self,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
        V_contain_lagrangian_degree: Optional[ContainmentLagrangianDegree],
        h_contain_lagrangian_degree: List[ContainmentLagrangianDegree],
        x_inner_init: np.ndarray,
        max_iter: int = 10,
        convergence_tol: float = 1e-3,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        trust_region: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Args:
          x_inner_init: The initial guess on a point inside V(x) <= 1 and
            h(x) >= 0. The initial ellipsoid will cover this point.
        """
        prog = solvers.MathematicalProgram()
        dim = self.x_set.size()

        S_ellipsoid = prog.NewSymmetricContinuousVariables(dim, "S")
        prog.AddPositiveSemidefiniteConstraint(S_ellipsoid)
        b_ellipsoid = prog.NewContinuousVariables(dim, "b")
        c_ellipsoid = prog.NewContinuousVariables(1, "c")[0]

        ellipsoid = sym.Polynomial(
            self.x.dot(S_ellipsoid @ self.x) + b_ellipsoid.dot(self.x) + c_ellipsoid,
            self.x_set,
        )
        prog.AddIndeterminates(self.x_set)
        if V_contain_lagrangian_degree is not None:
            V_contain_lagrangian = V_contain_lagrangian_degree.construct_lagrangian(
                prog, self.x_set
            )
            assert V is not None
            V_contain_lagrangian.add_constraint(
                prog,
                inner_ineq_poly=np.array([ellipsoid]),
                inner_eq_poly=self.state_eq_constraints,
                outer_poly=V - 1,
            )
        h_contain_lagrangians = [
            degree.construct_lagrangian(prog, self.x_set)
            for degree in h_contain_lagrangian_degree
        ]

        for i in range(len(h_contain_lagrangians)):
            h_contain_lagrangians[i].add_constraint(
                prog,
                inner_ineq_poly=np.array([ellipsoid]),
                inner_eq_poly=self.state_eq_constraints,
                outer_poly=-h[i],
            )

        # Make sure x_inner_init is inside V(x) <= 1 and h(x) >= 0.
        env_inner_init = {self.x[i]: x_inner_init[i] for i in range(self.nx)}
        if V is not None:
            assert V.Evaluate(env_inner_init) <= 1
        for h_i in h:
            assert h_i.Evaluate(env_inner_init) >= 0

        # First solve an optimization problem to find an inner ellipsoid.
        # Add a constraint that the initial ellipsoid contains x_inner_init.
        x_inner_init_in_ellipsoid = (
            ellipsoid_utils.add_ellipsoid_contain_pts_constraint(
                prog,
                S_ellipsoid,
                b_ellipsoid,
                c_ellipsoid,
                x_inner_init.reshape((1, -1)),
            )
        )
        result_init = solve_with_id(prog, solver_id, None)
        assert result_init.is_success()
        S_ellipsoid_init = result_init.GetSolution(S_ellipsoid)
        b_ellipsoid_init = result_init.GetSolution(b_ellipsoid)
        c_ellipsoid_init = result_init.GetSolution(c_ellipsoid)
        prog.RemoveConstraint(x_inner_init_in_ellipsoid)

        S_sol, b_sol, c_sol = ellipsoid_utils.maximize_inner_ellipsoid_sequentially(
            prog,
            S_ellipsoid,
            b_ellipsoid,
            c_ellipsoid,
            S_ellipsoid_init,
            b_ellipsoid_init,
            c_ellipsoid_init,
            max_iter,
            convergence_tol,
            solver_id,
            solver_options,
            trust_region,
        )
        return (S_sol, b_sol, c_sol)

    def _add_ellipsoid_in_compatible_region_constraint(
        self,
        prog: solvers.MathematicalProgram,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
        S_ellipsoid_inner: np.ndarray,
        b_ellipsoid_inner: np.ndarray,
        c_ellipsoid_inner: float,
    ):
        """
        Add the constraint that the ellipsoid is contained within the
        compatible region {x | V(x) <= 1, h(x) >= 0}.
        """
        ellipsoid_poly = sym.Polynomial(
            self.x.dot(S_ellipsoid_inner @ self.x)
            + b_ellipsoid_inner.dot(self.x)
            + c_ellipsoid_inner,
            self.x_set,
        )
        if V is not None:
            V_degree = V.TotalDegree()
            inner_eq_lagrangian_degree = (
                []
                if self.state_eq_constraints is None
                else [
                    V_degree - poly.TotalDegree() for poly in self.state_eq_constraints
                ]
            )
            ellipsoid_in_V_lagrangian_degree = ContainmentLagrangianDegree(
                inner_ineq=[V_degree - 2], inner_eq=inner_eq_lagrangian_degree, outer=-1
            )
            ellipsoid_in_V_lagrangian = (
                ellipsoid_in_V_lagrangian_degree.construct_lagrangian(prog, self.x_set)
            )
            ellipsoid_in_V_lagrangian.add_constraint(
                prog,
                inner_ineq_poly=np.array([ellipsoid_poly]),
                inner_eq_poly=self.state_eq_constraints,
                outer_poly=V - sym.Polynomial({sym.Monomial(): sym.Expression(1)}),
            )
        for i in range(h.size):
            h_degree = h[i].TotalDegree()
            inner_eq_lagrangian_degree = (
                []
                if self.state_eq_constraints is None
                else [
                    h_degree - poly.TotalDegree() for poly in self.state_eq_constraints
                ]
            )
            ellipsoid_in_h_lagrangian_degree = ContainmentLagrangianDegree(
                inner_ineq=[h_degree - 2], inner_eq=inner_eq_lagrangian_degree, outer=-1
            )
            ellipsoid_in_h_lagrangian = (
                ellipsoid_in_h_lagrangian_degree.construct_lagrangian(prog, self.x_set)
            )
            ellipsoid_in_h_lagrangian.add_constraint(
                prog,
                inner_ineq_poly=np.array([ellipsoid_poly]),
                inner_eq_poly=self.state_eq_constraints,
                outer_poly=-h[i],
            )

    def _add_compatible_states_options(
        self,
        prog: solvers.MathematicalProgram,
        V: Optional[sym.Polynomial],
        h: np.ndarray,
        compatible_states_options: CompatibleStatesOptions,
    ):
        compatible_states_options.add_cost(prog, self.x, V, h)
        compatible_states_options.add_constraint(prog, self.x, h)

    def _get_V_contain_ellipsoid_lagrangian_degree(
        self, V: Optional[sym.Polynomial]
    ) -> Optional[ContainmentLagrangianDegree]:
        if V is None:
            return None
        else:
            return ContainmentLagrangianDegree(
                inner_ineq=[-1],
                inner_eq=(
                    []
                    if self.state_eq_constraints is None
                    else [
                        np.maximum(0, V.TotalDegree() - poly.TotalDegree())
                        for poly in self.state_eq_constraints
                    ]
                ),
                outer=0,
            )

    def _get_h_contain_ellipsoid_lagrangian_degrees(
        self, h: np.ndarray
    ) -> List[ContainmentLagrangianDegree]:
        return [
            ContainmentLagrangianDegree(
                inner_ineq=[-1],
                inner_eq=(
                    []
                    if self.state_eq_constraints is None
                    else [
                        np.maximum(0, h_i.TotalDegree() - poly.TotalDegree())
                        for poly in self.state_eq_constraints
                    ]
                ),
                outer=0,
            )
            for h_i in h
        ]


def save_clf_cbf(
    V: Optional[sym.Polynomial],
    h: np.ndarray,
    x_set: sym.Variables,
    kappa_V: Optional[float],
    kappa_h: np.ndarray,
    pickle_path: str,
):
    """
    Save the CLF and CBF to a pickle file.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}
    if V is not None:
        data["V"] = compatible_clf_cbf.utils.serialize_polynomial(V, x_set)
    data["h"] = [compatible_clf_cbf.utils.serialize_polynomial(h_i, x_set) for h_i in h]
    if kappa_V is not None:
        data["kappa_V"] = kappa_V
    data["kappa_h"] = kappa_h

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


def load_clf_cbf(pickle_path: str, x_set: sym.Variables) -> dict:
    ret = {}
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    if "V" in data.keys():
        ret["V"] = compatible_clf_cbf.utils.deserialize_polynomial(data["V"], x_set)
    ret["h"] = np.array(
        [
            compatible_clf_cbf.utils.deserialize_polynomial(h_i, x_set)
            for h_i in data["h"]
        ]
    )
    if "kappa_V" in data.keys():
        ret["kappa_V"] = data["kappa_V"]
    ret["kappa_h"] = data["kappa_h"]
    return ret
