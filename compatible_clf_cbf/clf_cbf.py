from dataclasses import dataclass
from typing import List, Optional, Tuple
from typing_extensions import Self

import numpy as np

import pydrake.symbolic as sym
import pydrake.solvers as solvers

from compatible_clf_cbf.utils import (
    ContainmentLagrangianDegree,
    check_array_of_polynomials,
    get_polynomial_result,
    new_sos_polynomial,
    solve_with_id,
)
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
    # The Lagrangian polynomials multiplies with b(x)+ε. Should be an array of SOS
    # polynomials.
    b_plus_eps: Optional[np.ndarray]
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
        b_plus_eps_result = (
            get_polynomial_result(result, self.b_plus_eps, coefficient_tol)
            if self.b_plus_eps is not None
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
            b_plus_eps=b_plus_eps_result,
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
        ) -> sym.Polynomial:
            """
            Args:
              is_sos: whether the constructed polynomial is sos or not.
            """
            if is_sos:
                basis = sym.MonomialBasis(
                    {x: int(np.floor(self.x / 2)), y: int(np.floor(self.y / 2))}
                )
                poly, _ = prog.NewSosPolynomial(basis)
            else:
                basis = sym.MonomialBasis({x: self.x, y: self.y})
                coeffs = prog.NewContinuousVariables(basis.size)
                poly = sym.Polynomial({basis[i]: coeffs[i] for i in range(basis.size)})
            return poly

    lambda_y: List[Degree]
    xi_y: Degree
    y: Optional[List[Degree]]
    rho_minus_V: Optional[Degree]
    b_plus_eps: Optional[List[Degree]]
    state_eq_constraints: Optional[List[Degree]]

    def to_lagrangians(
        self, prog: solvers.MathematicalProgram, x: sym.Variables, y: sym.Variables
    ) -> CompatibleLagrangians:
        lambda_y = np.array(
            [
                lambda_y_i.construct_polynomial(prog, x, y, is_sos=False)
                for lambda_y_i in self.lambda_y
            ]
        )
        xi_y = self.xi_y.construct_polynomial(prog, x, y, is_sos=False)
        y_lagrangian = (
            None
            if self.y is None
            else np.array(
                [y_i.construct_polynomial(prog, x, y, is_sos=True) for y_i in self.y]
            )
        )
        rho_minus_V = (
            None
            if self.rho_minus_V is None
            else self.rho_minus_V.construct_polynomial(prog, x, y, is_sos=True)
        )
        b_plus_eps = (
            None
            if self.b_plus_eps is None
            else np.array(
                [
                    b_plus_eps_i.construct_polynomial(prog, x, y, is_sos=True)
                    for b_plus_eps_i in self.b_plus_eps
                ]
            )
        )
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
        return CompatibleLagrangians(
            lambda_y=lambda_y,
            xi_y=xi_y,
            y=y_lagrangian,
            rho_minus_V=rho_minus_V,
            b_plus_eps=b_plus_eps,
            state_eq_constraints=state_eq_constraints,
        )


@dataclass
class UnsafeRegionLagrangians:
    """
    The Lagrangians for certifying that the 0-super level set of a CBF doesn't
    intersect with an unsafe region.

    For a CBF function bᵢ(x), to prove that the 0-super level set
    {x |bᵢ(x) >= 0} doesn't intersect with an unsafe set
    {x | pⱼ(x) <= 0 for all j}, we impose the condition:

    -(1+ϕᵢ,₀(x))*bᵢ(x) +∑ⱼϕᵢ,ⱼ(x)pⱼ(x) is sos
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
        return UnsafeRegionLagrangians(
            cbf=get_polynomial_result(result, self.cbf, coefficient_tol),
            unsafe_region=get_polynomial_result(
                result, self.unsafe_region, coefficient_tol
            ),
            state_eq_constraints=None
            if self.state_eq_constraints is None
            else get_polynomial_result(
                result, self.state_eq_constraints, coefficient_tol
            ),
        )


@dataclass
class UnsafeRegionLagrangianDegrees:
    cbf: int
    unsafe_region: List[int]
    state_eq_constraints: Optional[List[int]]

    def to_lagrangians(
        self, prog: solvers.MathematicalProgram, x_set: sym.Variables
    ) -> UnsafeRegionLagrangians:
        cbf, _ = new_sos_polynomial(prog, x_set, self.cbf)

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
        return UnsafeRegionLagrangians(
            cbf=cbf,
            unsafe_region=unsafe_region,
            state_eq_constraints=state_eq_constraints,
        )


class CompatibleClfCbf:
    """
    Certify and synthesize compatible Control Lyapunov Function (CLF) and
    Control Barrier Functions (CBFs).

    For a continuous-time control-affine system
    ẋ = f(x)+g(x)u, u∈𝒰
    A CLF V(x) and a CBF b(x) is compatible if and only if
    ∃ u∈𝒰,      ∂b/∂x*f(x) + ∂b/∂x*g(x)*u ≥ −κ_b*b(x)
            and ∂V/∂x*f(x) + ∂V/∂x*g(x)*u ≤ −κ_V*V(x)
    For simplicity, let's first consider that u is un-constrained, namely 𝒰 is
    the entire space.
    By Farkas lemma, this is equivalent to the following set being empty

    {(x, y) | [y(0)]ᵀ*[-∂b/∂x*g(x)] = 0, [y(0)]ᵀ*[ ∂b/∂x*f(x)+κ_b*b(x)] = -1, y>=0}        (1)
              [y(1)]  [ ∂V/∂x*g(x)]      [y(1)]  [-∂V/∂x*f(x)-κ_V*V(x)]

    We can then use Positivstellensatz to certify the emptiness of this set.

    The same math applies to multiple CBFs, or when u is constrained within a
    polyhedron.
    """  # noqa E501

    def __init__(
        self,
        *,
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        unsafe_regions: List[np.ndarray],
        Au: Optional[np.ndarray] = None,
        bu: Optional[np.ndarray] = None,
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
          unsafe_regions: List[np.ndarray]
            A list of numpy arrays of polynomials. unsafe_regions[i] is the i'th
            unsafe region (to be certified by the i'th CBF). The i'th unsafe
            region is the 0-sublevel set of unsafe_regions[i].
          Au: Optional[np.ndarray]
            The set of admissible control is Au * u <= bu.
            The shape is (Any, nu)
          bu: Optional[np.ndarray]
            The set of admissible control is Au * u <= bu.
            The shape is (Any,)
          with_clf: bool
            Whether to certify or search for CLF. If set to False, then we will
            certify or search multiple compatible CBFs without CLF.
          use_y_squared: bool
            For that empty set in the class documentation, we could replace
            y>=0 condition with using y². This will potentially reduce the
            number of Lagrangian multipliers in the p-satz, but increase the
            total degree of the polynomials. Set use_y_squared=True if we use
            y², and we certify the set

            {(x, y) | [y(0)²]ᵀ*[-∂b/∂x*g(x)] = 0, [y(0)²]ᵀ*[ ∂b/∂x*f(x)+κ_b*b(x)] = -1}       (2)
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
        for unsafe_region in unsafe_regions:
            check_array_of_polynomials(unsafe_region, self.x_set)
        self.unsafe_regions = unsafe_regions
        if Au is not None:
            assert Au.shape[1] == self.nu
            assert bu is not None
            assert bu.shape == (Au.shape[0],)
        self.Au = Au
        self.bu = bu
        self.with_clf = with_clf
        self.use_y_squared = use_y_squared
        y_size = len(self.unsafe_regions) + (1 if self.with_clf else 0)
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

    def certify_cbf_unsafe_region(
        self,
        unsafe_region_index: int,
        cbf: sym.Polynomial,
        lagrangian_degrees: UnsafeRegionLagrangianDegrees,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
    ) -> Optional[UnsafeRegionLagrangians]:
        """
        Certifies that the 0-superlevel set {x | bᵢ(x) >= 0} does not intersect
        with the unsafe region self.unsafe_regions[unsafe_region_index].

        If we denote the unsafe region as {x | p(x) <= 0}, then we impose the constraint

        We impose the constraint
        -(1+ϕᵢ,₀(x))*bᵢ(x) +∑ⱼϕᵢ,ⱼ(x)pⱼ(x) is sos
        ϕᵢ,₀(x), ϕᵢ,ⱼ(x) are sos.

        Args:
          unsafe_region_index: We certify the CBF for the region
            self.unsafe_regions[unsafe_region_index]
          cbf: bᵢ(x) in the documentation above. The CBF function for
            self.unsafe_regions[unsafe_region_index]
        """
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x_set)
        lagrangians = lagrangian_degrees.to_lagrangians(prog, self.x_set)
        self._add_barrier_safe_constraint(prog, unsafe_region_index, cbf, lagrangians)
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
        b: np.ndarray,
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
        lagrangian_degrees: CompatibleLagrangianDegrees,
        barrier_eps: Optional[np.ndarray],
        local_clf: bool = True,
    ) -> Tuple[solvers.MathematicalProgram, CompatibleLagrangians]:
        """
        Given CLF candidate V and CBF candidate b, construct the optimization
        program to certify that they are compatible within the region
        {x | V(x) <= 1} ∩ {x | b(x) >= -eps}.

        Args:
          V: The CLF candidate. If empty, then we will certify that the multiple
            barrier functions are compatible.
          b: The CBF candidates.
          kappa_V: The exponential decay rate for CLF. Namely we want V̇ ≤ −κ_V*V
          kappa_b: The exponential rate for CBF, namely we want ḃ ≥ −κ_b*b
          lagrangian_degrees: The degrees for the Lagrangian polynomials.
          barrier_eps: The certified safe region is {x | b(x) >= -eps}
          coefficient_tol: In the Lagrangian polynomials, we will remove the
            coefficients no larger than this tolerance.
          local_clf: Whether the CLF is valid in a local region or globally.
        Returns:
          result: The result for solving the optimization program.
          lagrangian_result: The result of the Lagrangian polynomials.
        """
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)
        lagrangians = lagrangian_degrees.to_lagrangians(prog, self.x_set, self.y_set)
        self._add_compatibility(
            prog=prog,
            V=V,
            b=b,
            kappa_V=kappa_V,
            kappa_b=kappa_b,
            lagrangians=lagrangians,
            barrier_eps=barrier_eps,
            local_clf=local_clf,
        )
        return (prog, lagrangians)

    def search_lagrangians_given_clf_cbf(
        self,
        V: Optional[sym.Polynomial],
        b: np.ndarray,
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
        barrier_eps: np.ndarray,
        compatible_lagrangian_degrees: CompatibleLagrangianDegrees,
        unsafe_regions_lagrangian_degrees: List[UnsafeRegionLagrangianDegrees],
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
    ) -> Tuple[
        Optional[CompatibleLagrangians], List[Optional[UnsafeRegionLagrangians]]
    ]:
        (
            prog_compatible,
            compatible_lagrangians,
        ) = self.construct_search_compatible_lagrangians(
            V, b, kappa_V, kappa_b, compatible_lagrangian_degrees, barrier_eps
        )
        result_compatible = solve_with_id(prog_compatible, solver_id, solver_options)
        compatible_lagrangians_result = (
            compatible_lagrangians.get_result(
                result_compatible, lagrangian_coefficient_tol
            )
            if result_compatible.is_success()
            else None
        )
        unsafe_lagrangians_result: List[Optional[UnsafeRegionLagrangians]] = [
            None
        ] * len(self.unsafe_regions)
        for i in range(len(self.unsafe_regions)):
            unsafe_lagrangians_result[i] = self.certify_cbf_unsafe_region(
                i,
                b[i],
                unsafe_regions_lagrangian_degrees[i],
                solver_id,
                solver_options,
                lagrangian_coefficient_tol,
            )
        return compatible_lagrangians_result, unsafe_lagrangians_result

    def search_clf_cbf_given_lagrangian(
        self,
        compatible_lagrangians: CompatibleLagrangians,
        unsafe_regions_lagrangians: List[UnsafeRegionLagrangians],
        clf_degree: Optional[int],
        cbf_degrees: List[int],
        x_equilibrium: Optional[np.ndarray],
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
        barrier_eps: np.ndarray,
        S_ellipsoid_inner: np.ndarray,
        b_ellipsoid_inner: np.ndarray,
        c_ellipsoid_inner: float,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
    ) -> Tuple[
        Optional[sym.Polynomial],
        Optional[np.ndarray],
        solvers.MathematicalProgramResult,
    ]:
        """
        Given the Lagrangian multipliers and an inner ellipsoid, find the clf
        and cbf, such that the compatible region contains that inner ellipsoid.

        Returns: (V, b, result)
          V: The CLF.
          b: The CBF.
          result: The result of the optimization program.
        """
        prog, V, b = self._construct_search_clf_cbf_program(
            compatible_lagrangians,
            unsafe_regions_lagrangians,
            clf_degree,
            cbf_degrees,
            x_equilibrium,
            kappa_V,
            kappa_b,
            barrier_eps,
        )

        self._add_ellipsoid_in_compatible_region_constraint(
            prog, V, b, S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner
        )

        result = solve_with_id(prog, solver_id, solver_options)
        if result.is_success():
            V_sol = None if V is None else result.GetSolution(V)
            b_sol = np.array([result.GetSolution(b_i) for b_i in b])
        else:
            V_sol = None
            b_sol = None
        return V_sol, b_sol, result

    def binary_search_clf_cbf(
        self,
        compatible_lagrangians: CompatibleLagrangians,
        unsafe_regions_lagrangians: List[UnsafeRegionLagrangians],
        clf_degree: Optional[int],
        cbf_degrees: List[int],
        x_equilibrium: Optional[np.ndarray],
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
        barrier_eps: np.ndarray,
        S_ellipsoid_inner: np.ndarray,
        b_ellipsoid_inner: np.ndarray,
        c_ellipsoid_inner: float,
        scale_min: float,
        scale_max: float,
        scale_tol: float,
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
          scale_min: The minimum of the ellipsoid scaling factor.
          scale_max: The maximal of the ellipsoid scaling factor.
          scale_tol: Terminate the binary search when the difference between
            the max/min scaling factor is below this tolerance.

        Return: (V, b)
        """

        def search(
            scale,
        ) -> Tuple[
            Optional[sym.Polynomial],
            Optional[np.ndarray],
            solvers.MathematicalProgramResult,
        ]:
            c_new = ellipsoid_utils.scale_ellipsoid(
                S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner, scale
            )
            V, b, result = self.search_clf_cbf_given_lagrangian(
                compatible_lagrangians,
                unsafe_regions_lagrangians,
                clf_degree,
                cbf_degrees,
                x_equilibrium,
                kappa_V,
                kappa_b,
                barrier_eps,
                S_ellipsoid_inner,
                b_ellipsoid_inner,
                c_new,
                solver_id,
                solver_options,
            )
            return V, b, result

        assert scale_max >= scale_min
        assert scale_tol > 0
        V, b, result = search(scale_max)
        if result.is_success():
            print(f"binary_search_clf_cbf: scale={scale_max} is feasible.")
            assert b is not None
            return V, b

        V_success, b_success, result = search(scale_min)
        assert (
            result.is_success()
        ), f"binary_search_clf_cbf: scale_min={scale_min} is not feasible."
        assert b_success is not None

        while scale_max - scale_min > scale_tol:
            scale = (scale_max + scale_min) / 2
            V, b, result = search(scale)
            if result.is_success():
                print(f"binary_search_clf_cbf: scale={scale} is feasible.")
                scale_min = scale
                V_success = V
                assert b is not None
                b_success = b
            else:
                print(f"binary_search_clf_cbf: scale={scale} is not feasible.")
                scale_max = scale

        return V_success, b_success

    def in_compatible_region(
        self,
        V: Optional[sym.Polynomial],
        b: np.ndarray,
        x_samples: np.ndarray,
    ) -> np.ndarray:
        """
        Returns if x_samples[i] is in the compatible region
        {x | V(x) <= 1, b(x) >= 0}.

        Return:
        in_compatible_flag: in_compatible_flag[i] is True iff x_samples[i] is
          in the compatible region.
        """
        in_b = np.all(
            np.concatenate(
                [
                    (b_i.EvaluateIndeterminates(self.x, x_samples.T) >= 0).reshape(
                        (-1, 1)
                    )
                    for b_i in b
                ],
                axis=1,
            ),
            axis=1,
        )
        if V is not None:
            in_V = V.EvaluateIndeterminates(self.x, x_samples.T) <= 1
            return np.logical_and(in_b, in_V)
        else:
            return in_b

    def bilinear_alternation(
        self,
        V_init: Optional[sym.Polynomial],
        b_init: np.ndarray,
        compatible_lagrangian_degrees: CompatibleLagrangianDegrees,
        unsafe_regions_lagrangian_degrees: List[UnsafeRegionLagrangianDegrees],
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
        barrier_eps: np.ndarray,
        x_equilibrium: np.ndarray,
        clf_degree: Optional[int],
        cbf_degrees: List[int],
        max_iter: int,
        *,
        x_inner: np.ndarray,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
        ellipsoid_trust_region: Optional[float] = 100,
        binary_search_scale_min: float = 1,
        binary_search_scale_max: float = 2,
        binary_search_scale_tol: float = 0.1,
        find_inner_ellipsoid_max_iter: int = 3,
    ) -> Tuple[Optional[sym.Polynomial], np.ndarray]:
        """
        Synthesize the compatible CLF and CBF through bilinear alternation. We
        alternate between
        1. Fixing the CLF/CBF, searching for Lagrangians.
        2. Fixing Lagrangians, searching for CLF/CBF.

        Args:
          x_inner: Each row of x_inner is a state that should be contained in
            the compatible region.
          max_iter: The maximal number of bilinear alternation iterations.
          lagrangian_coefficient_tol: We remove the coefficients whose absolute
            value is smaller than this tolerance in the Lagrangian polynomials.
            Use None to preserve all coefficients.
          ellipsoid_trust_region: when we search for the ellipsoid, we put a
            trust region constraint. This is the squared radius of that trust
            region.
        """

        iteration = 0
        clf = V_init
        assert len(b_init) == len(self.unsafe_regions)
        cbf = b_init

        compatible_lagrangians = None
        unsafe_lagrangians: List[Optional[UnsafeRegionLagrangians]] = [None] * len(
            self.unsafe_regions
        )

        for iteration in range(max_iter):
            print(f"iteration {iteration}")
            # Search for the Lagrangians.
            (
                compatible_lagrangians,
                unsafe_lagrangians,
            ) = self.search_lagrangians_given_clf_cbf(
                clf,
                cbf,
                kappa_V,
                kappa_b,
                barrier_eps,
                compatible_lagrangian_degrees,
                unsafe_regions_lagrangian_degrees,
                solver_id,
                solver_options,
                lagrangian_coefficient_tol,
            )
            assert compatible_lagrangians is not None
            assert all(unsafe_lagrangians)

            # Search for the inner ellipsoid.
            V_contain_ellipsoid_lagrangian_degree = (
                self._get_V_contain_ellipsoid_lagrangian_degree(clf)
            )
            b_contain_ellipsoid_lagrangian_degree = (
                self._get_b_contain_ellipsoid_lagrangian_degrees(cbf)
            )
            (
                S_ellipsoid_inner,
                b_ellipsoid_inner,
                c_ellipsoid_inner,
            ) = self._find_max_inner_ellipsoid(
                clf,
                cbf,
                V_contain_ellipsoid_lagrangian_degree,
                b_contain_ellipsoid_lagrangian_degree,
                x_inner,
                solver_id=solver_id,
                max_iter=find_inner_ellipsoid_max_iter,
                trust_region=ellipsoid_trust_region,
            )

            clf, cbf = self.binary_search_clf_cbf(
                compatible_lagrangians,
                unsafe_lagrangians,
                clf_degree,
                cbf_degrees,
                x_equilibrium,
                kappa_V,
                kappa_b,
                barrier_eps,
                S_ellipsoid_inner,
                b_ellipsoid_inner,
                c_ellipsoid_inner,
                binary_search_scale_min,
                binary_search_scale_max,
                binary_search_scale_tol,
                solver_id,
                solver_options,
            )
        return clf, cbf

    def check_compatible_at_state(
        self,
        V: Optional[sym.Polynomial],
        b: np.ndarray,
        x_val: np.ndarray,
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
    ) -> Tuple[bool, solvers.MathematicalProgramResult]:
        """
        Check if at a given state the CLF and CBFs are compatible, namely there
        exists a common u such that
        Vdot(x, u) <= -kappa_V * V
        bdot(x, u) >= -kappa_b * b
        """
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(self.nu, "u")
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
        for b_i in b:
            b_val = b_i.Evaluate(env)
            dbdx = b_i.Jacobian(self.x)
            dbdx_val = np.array([dbdx[i].Evaluate(env) for i in range(self.nx)])
            dbdx_times_f_val = dbdx_val.dot(f_val)
            dbdx_times_g_val = dbdx_val @ g_val
            prog.AddLinearConstraint(
                dbdx_times_f_val + dbdx_times_g_val @ u >= -kappa_b * b_val
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
        b: np.ndarray,
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute
        Λ(x) = [-∂b/∂x*g(x)]
               [ ∂V/∂x*g(x)]
        ξ(x) = [ ∂b/∂x*f(x)+κ_b*b(x)]
               [-∂V/∂x*f(x)-κ_V*V(x)]

        Args:
          V: The CLF function. If with_clf is False, then V is None.
          b: An array of CBFs. b[i] is the CBF for the i'th unsafe region.
          kappa_V: κ_V in the documentation above.
          kappa_b: κ_b in the documentation above. kappa_b[i] is the kappa for b[i].
        Returns:
          (xi, lambda_mat) ξ(x) and Λ(x) in the documentation above.
        """
        num_unsafe_regions = len(self.unsafe_regions)
        if self.with_clf:
            assert isinstance(V, sym.Polynomial)
            dVdx = V.Jacobian(self.x)
            xi_rows = num_unsafe_regions + 1
        else:
            assert V is None
            dVdx = None
            xi_rows = num_unsafe_regions
        assert b.shape == (len(self.unsafe_regions),)
        assert kappa_b.shape == b.shape
        dbdx = np.concatenate(
            [b[i].Jacobian(self.x).reshape((1, -1)) for i in range(b.size)], axis=0
        )
        lambda_mat = np.empty((xi_rows, self.nu), dtype=object)
        lambda_mat[:num_unsafe_regions] = -dbdx @ self.g
        xi = np.empty((xi_rows,), dtype=object)
        xi[:num_unsafe_regions] = dbdx @ self.f + kappa_b * b
        # TODO(hongkai.dai): support input bounds Au * u <= bu
        assert self.Au is None and self.bu is None

        if self.with_clf:
            lambda_mat[-1] = dVdx @ self.g
            xi[-1] = -dVdx.dot(self.f) - kappa_V * V

        return (xi, lambda_mat)

    def _add_compatibility(
        self,
        *,
        prog: solvers.MathematicalProgram,
        V: Optional[sym.Polynomial],
        b: np.ndarray,
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
        lagrangians: CompatibleLagrangians,
        barrier_eps: Optional[np.ndarray],
        local_clf: bool,
    ) -> sym.Polynomial:
        """
        Add the p-satz condition that certifies the following set is empty
        if use_y_squared = False:
        {(x, y) | [y(0)]ᵀ*[-∂b/∂x*g(x)] = [0], [y(0)]ᵀ*[ ∂b/∂x*f(x)+κ_b*b(x)] = -1, y>=0, V(x)≤ρ, b(x)≥−ε}         (1)
                  [y(1)]  [ ∂V/∂x*g(x)]   [0]  [y(1)]  [-∂V/∂x*f(x)-κ_V*V(x)]
        if use_y_squared = True:
        {(x, y) | [y(0)²]ᵀ*[-∂b/∂x*g(x)] = [0], [y(0)²]ᵀ*[ ∂b/∂x*f(x)+κ_b*b(x)] = -1, V(x)≤ρ, b(x)≥−ε}              (2)
                  [y(1)²]  [ ∂V/∂x*g(x)]   [0]  [y(1)²]  [-∂V/∂x*f(x)-κ_V*V(x)]
        namely inside the set {x | V(x)≤ρ, b(x)≥−ε}, the CLF and CBF are compatible.

        Let's denote
        Λ(x) = [-∂b/∂x*g(x)]
               [ ∂V/∂x*g(x)]
        ξ(x) = [ ∂b/∂x*f(x)+κ_b*b(x)]
               [-∂V/∂x*f(x)-κ_V*V(x)]
        To certify the emptiness of the set in (1), we can use the sufficient condition
        -1 - s₀(x, y)ᵀ Λ(x)ᵀy - s₁(x, y)(ξ(x)ᵀy+1) - s₂(x, y)ᵀy - s₃(x, y)(1 − V) - s₄(x, y)ᵀ(b(x)+ε) is sos          (3)
        s₂(x, y), s₃(x, y), s₄(x, y) are all sos.

        To certify the emptiness of the set in (2), we can use the sufficient condition
        -1 - s₀(x, y)ᵀ Λ(x)ᵀy² - s₁(x, y)(ξ(x)ᵀy²+1) - s₃(x, y)(1 − V) - s₄(x, y)ᵀ(b(x)+ε) is sos                     (4)
        s₃(x, y), s₄(x, y) are all sos.

        Note that we do NOT add the constraint
        s₂(x, y), s₃(x, y), s₄(x, y) are all sos.
        in this function. The user should add this constraint separately.

        Returns:
          poly: The polynomial on the left hand side of equation (3) or (4).
        """  # noqa: E501
        xi, lambda_mat = self._calc_xi_Lambda(
            V=V, b=b, kappa_V=kappa_V, kappa_b=kappa_b
        )
        poly_one = sym.Polynomial(sym.Monomial())

        poly = -poly_one
        # Compute s₀(x, y)ᵀ Λ(x)ᵀy
        if self.use_y_squared:
            lambda_y = lambda_mat.T @ self.y_squared_poly
        else:
            lambda_y = lambda_mat.T @ self.y_poly
        poly -= lagrangians.lambda_y.dot(lambda_y)

        # Compute s₁(x, y)(ξ(x)ᵀy+1)
        # This is just polynomial 1.
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

        # Compute s₄(x, y)ᵀ(b(x)+ε)
        if barrier_eps is not None:
            assert np.all(barrier_eps >= 0)
            assert lagrangians.b_plus_eps is not None
            poly -= lagrangians.b_plus_eps.dot(barrier_eps + b)

        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)

        prog.AddSosConstraint(poly)
        return poly

    def _add_barrier_safe_constraint(
        self,
        prog: solvers.MathematicalProgram,
        unsafe_region_index: int,
        b: sym.Polynomial,
        lagrangians: UnsafeRegionLagrangians,
    ) -> sym.Polynomial:
        """
        Adds the constraint that the 0-superlevel set of the barrier function
        does not intersect with the unsafe region.
        Since the i'th unsafe regions is defined as the 0-sublevel set of
        polynomials p(x), we want to certify that the set {x|p(x)≤0, bᵢ(x)≥0}
        is empty.
        The emptiness of the set can be certified by the constraint
        -(1+ϕᵢ,₀(x))bᵢ(x) +∑ⱼϕᵢ,ⱼ(x)pⱼ(x) is sos
        ϕᵢ,₀(x), ϕᵢ,ⱼ(x) are sos.

        Note that this function only adds the constraint
        -(1+ϕᵢ,₀(x))*bᵢ(x) +∑ⱼϕᵢ,ⱼ(x)pⱼ(x) is sos
        It doesn't add the constraint ϕᵢ,₀(x), ϕᵢ,ⱼ(x) are sos.

        Args:
          unsafe_region_index: We certify that the 0-superlevel set of the
            barrier function doesn't intersect with the unsafe region
            self.unsafe_regions[unsafe_region_index]
          b: a polynomial, b is the barrier function for the
            unsafe region self.unsafe_regions[unsafe_region_index].
          lagrangians: A array of polynomials, ϕᵢ(x) in the documentation above.
        Returns:
          poly: poly is the polynomial -(1+ϕᵢ,₀(x))bᵢ(x) + ∑ⱼϕᵢ,ⱼ(x)pⱼ(x)
        """
        assert lagrangians.unsafe_region.size == len(
            self.unsafe_regions[unsafe_region_index]
        )
        poly = -(1 + lagrangians.cbf) * b + lagrangians.unsafe_region.dot(
            self.unsafe_regions[unsafe_region_index]
        )
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
        prog.AddSosConstraint(poly)
        return poly

    def _construct_search_clf_cbf_program(
        self,
        compatible_lagrangians: CompatibleLagrangians,
        unsafe_regions_lagrangians: List[UnsafeRegionLagrangians],
        clf_degree: Optional[int],
        cbf_degrees: List[int],
        x_equilibrium: Optional[np.ndarray],
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
        barrier_eps: np.ndarray,
        local_clf: bool = True,
    ) -> Tuple[solvers.MathematicalProgram, Optional[sym.Polynomial], np.ndarray,]:
        """
        Construct a program to search for compatible CLF/CBFs given the Lagrangians.
        Notice that we have not imposed the cost to the program yet.

        Args:
          compatible_lagrangians: The Lagrangian polynomials. Result from
            solving construct_search_compatible_lagrangians().
          unsafe_regions_lagrangians: unsafe_regions_lagrangians[i] is the
            Lagrangian polynomial for the i'th CBF to certify that the CBF
            0-super level set doesn't intersect with the i'th unsafe region.
          clf_degree: if not None, the total degree of CLF.
          cbf_degrees: cbf_degrees[i] is the total degree of the i'th CBF.
          x_equilibrium: if not None, the equilibrium state.
        """
        assert len(unsafe_regions_lagrangians) == len(self.unsafe_regions)
        assert len(cbf_degrees) == len(self.unsafe_regions)
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
                    -V_x_equilibrium_coeff[0],
                    V_x_equilibrium_var,
                )
        else:
            V = None

        # Add CBF.
        b = np.array(
            [
                prog.NewFreePolynomial(self.x_set, cbf_degree)
                for cbf_degree in cbf_degrees
            ]
        )
        for i in range(len(self.unsafe_regions)):
            self._add_barrier_safe_constraint(
                prog, i, b[i], unsafe_regions_lagrangians[i]
            )

        self._add_compatibility(
            prog=prog,
            V=V,
            b=b,
            kappa_V=kappa_V,
            kappa_b=kappa_b,
            lagrangians=compatible_lagrangians,
            barrier_eps=barrier_eps,
            local_clf=local_clf,
        )

        return (prog, V, b)

    def _find_max_inner_ellipsoid(
        self,
        V: Optional[sym.Polynomial],
        b: np.ndarray,
        V_contain_lagrangian_degree: Optional[ContainmentLagrangianDegree],
        b_contain_lagrangian_degree: List[ContainmentLagrangianDegree],
        x_inner_init: np.ndarray,
        max_iter: int = 10,
        convergence_tol: float = 1e-3,
        solver_id: Optional[solvers.SolverId] = None,
        trust_region: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Args:
          x_inner_init: The initial guess on a point inside V(x) <= 1 and
            b(x) >= 0. The initial ellipsoid will cover this point.
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
        b_contain_lagrangians = [
            degree.construct_lagrangian(prog, self.x_set)
            for degree in b_contain_lagrangian_degree
        ]

        for i in range(len(b_contain_lagrangians)):
            b_contain_lagrangians[i].add_constraint(
                prog,
                inner_ineq_poly=np.array([ellipsoid]),
                inner_eq_poly=self.state_eq_constraints,
                outer_poly=-b[i],
            )

        # Make sure x_inner_init is inside V(x) <= 1 and b(x) >= 0.
        env_inner_init = {self.x[i]: x_inner_init[i] for i in range(self.nx)}
        if V is not None:
            assert V.Evaluate(env_inner_init) <= 1
        for b_i in b:
            assert b_i.Evaluate(env_inner_init) >= 0

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
            trust_region,
        )
        return (S_sol, b_sol, c_sol)

    def _add_ellipsoid_in_compatible_region_constraint(
        self,
        prog: solvers.MathematicalProgram,
        V: Optional[sym.Polynomial],
        b: np.ndarray,
        S_ellipsoid_inner: np.ndarray,
        b_ellipsoid_inner: np.ndarray,
        c_ellipsoid_inner: float,
    ):
        """
        Add the constraint that the ellipsoid is contained within the
        compatible region {x | V(x) <= 1, b(x) >= 0}.
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
        for i in range(b.size):
            b_degree = b[i].TotalDegree()
            inner_eq_lagrangian_degree = (
                []
                if self.state_eq_constraints is None
                else [
                    b_degree - poly.TotalDegree() for poly in self.state_eq_constraints
                ]
            )
            ellipsoid_in_b_lagrangian_degree = ContainmentLagrangianDegree(
                inner_ineq=[b_degree - 2], inner_eq=inner_eq_lagrangian_degree, outer=-1
            )
            ellipsoid_in_b_lagrangian = (
                ellipsoid_in_b_lagrangian_degree.construct_lagrangian(prog, self.x_set)
            )
            ellipsoid_in_b_lagrangian.add_constraint(
                prog,
                inner_ineq_poly=np.array([ellipsoid_poly]),
                inner_eq_poly=self.state_eq_constraints,
                outer_poly=-b[i],
            )

    def _get_V_contain_ellipsoid_lagrangian_degree(
        self, V: Optional[sym.Polynomial]
    ) -> Optional[ContainmentLagrangianDegree]:
        if V is None:
            return None
        else:
            return ContainmentLagrangianDegree(
                inner_ineq=[-1],
                inner_eq=[]
                if self.state_eq_constraints is None
                else [
                    np.maximum(0, V.TotalDegree() - poly.TotalDegree())
                    for poly in self.state_eq_constraints
                ],
                outer=0,
            )

    def _get_b_contain_ellipsoid_lagrangian_degrees(
        self, b: np.ndarray
    ) -> List[ContainmentLagrangianDegree]:
        return [
            ContainmentLagrangianDegree(
                inner_ineq=[-1],
                inner_eq=[]
                if self.state_eq_constraints is None
                else [
                    np.maximum(0, b_i.TotalDegree() - poly.TotalDegree())
                    for poly in self.state_eq_constraints
                ],
                outer=0,
            )
            for b_i in b
        ]
