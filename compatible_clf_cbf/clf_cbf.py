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

    def initialize_lagrangians(
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

    def get_result(self, result: solvers.MathematicalProgramResult) -> Self:
        return UnsafeRegionLagrangians(
            cbf=result.GetSolution(self.cbf),
            unsafe_region=np.array(
                [result.GetSolution(phi) for phi in self.unsafe_region]
            ),
            state_eq_constraints=None
            if self.state_eq_constraints is None
            else np.array(
                [result.GetSolution(poly) for poly in self.state_eq_constraints]
            ),
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
        cbf_lagrangian_degree: int,
        unsafe_region_lagrangian_degrees: List[int],
        state_eq_constraints_lagrangian_degrees: Optional[List[int]] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
    ) -> UnsafeRegionLagrangians:
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
          cbf_lagrangian_degree: The degree of the polynomial ϕᵢ,₀(x).
          unsafe_region_lagrangian_degrees: unsafe_region_lagrangian_degrees[j]
          is the degree of the polynomial ϕᵢ,ⱼ₊₁(x)
        """
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x_set)
        cbf_lagrangian, cbf_lagrangian_gram = prog.NewSosPolynomial(
            self.x_set, cbf_lagrangian_degree
        )
        unsafe_lagrangians = np.array(
            [
                prog.NewSosPolynomial(self.x_set, deg)[0]
                for deg in unsafe_region_lagrangian_degrees
            ]
        )
        if self.state_eq_constraints is not None:
            assert state_eq_constraints_lagrangian_degrees is not None
            state_eq_constraints_lagrangians = np.array(
                [
                    prog.NewFreePolynomial(self.x_set, deg)
                    for deg in state_eq_constraints_lagrangian_degrees
                ]
            )
        else:
            state_eq_constraints_lagrangians = None
        lagrangians = UnsafeRegionLagrangians(
            cbf_lagrangian,
            unsafe_lagrangians,
            state_eq_constraints=state_eq_constraints_lagrangians,
        )
        self._add_barrier_safe_constraint(prog, unsafe_region_index, cbf, lagrangians)
        result = solvers.Solve(prog, None, solver_options)
        assert result.is_success()
        return lagrangians.get_result(result)

    def construct_search_compatible_lagrangians(
        self,
        V: Optional[sym.Polynomial],
        b: np.ndarray,
        kappa_V: Optional[float],
        kappa_b: np.ndarray,
        lagrangian_degrees: CompatibleLagrangianDegrees,
        rho: Optional[float],
        barrier_eps: Optional[np.ndarray],
    ) -> Tuple[solvers.MathematicalProgram, CompatibleLagrangians]:
        """
        Given CLF candidate V and CBF candidate b, construct the optimization
        program to certify that they are compatible within the region
        {x | V(x) <= rho} ∩ {x | b(x) >= -eps}.

        Args:
          V: The CLF candidate. If empty, then we will certify that the multiple
            barrier functions are compatible.
          b: The CBF candidates.
          kappa_V: The exponential decay rate for CLF. Namely we want V̇ ≤ −κ_V*V
          kappa_b: The exponential rate for CBF, namely we want ḃ ≥ −κ_b*b
          lagrangian_degrees: The degrees for the Lagrangian polynomials.
          rho: The certified inner approximation of ROA is {x | V(x) <= rho}
          barrier_eps: The certified safe region is {x | b(x) >= -eps}
          coefficient_tol: In the Lagrangian polynomials, we will remove the
            coefficients no larger than this tolerance.
        Returns:
          result: The result for solving the optimization program.
          lagrangian_result: The result of the Lagrangian polynomials.
        """
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)
        lagrangians = lagrangian_degrees.initialize_lagrangians(
            prog, self.x_set, self.y_set
        )
        self._add_compatibility(
            prog=prog,
            V=V,
            b=b,
            kappa_V=kappa_V,
            kappa_b=kappa_b,
            lagrangians=lagrangians,
            rho=rho,
            barrier_eps=barrier_eps,
        )
        return (prog, lagrangians)

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
        Optional[float],
        solvers.MathematicalProgramResult,
    ]:
        """
        Given the Lagrangian multipliers and an inner ellipsoid, find the clf
        and cbf, such that the compatible region contains that inner ellipsoid.

        Returns: (V, b, rho, result)
          V: The CLF.
          b: The CBF.
          rho: The certified ROA is {x | V(x) <= rho}.
          result: The result of the optimization program.
        """
        prog, V, b, rho = self._construct_search_clf_cbf_program(
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
            prog, V, b, rho, S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner
        )

        result = solve_with_id(prog, solver_id, solver_options)
        if result.is_success():
            V_sol = result.GetSolution(V)
            b_sol = np.array([result.GetSolution(b_i) for b_i in b])
            rho_sol = result.GetSolution(rho)
        else:
            V_sol = None
            b_sol = None
            rho_sol = None
        return V_sol, b_sol, rho_sol, result

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
    ) -> Tuple[Optional[sym.Polynomial], np.ndarray, Optional[float]]:
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

        Return: (V, b, rho)
        """

        def search(
            scale,
        ) -> Tuple[
            Optional[sym.Polynomial],
            Optional[np.ndarray],
            Optional[float],
            solvers.MathematicalProgramResult,
        ]:
            c_new = ellipsoid_utils.scale_ellipsoid(
                S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner, scale
            )
            V, b, rho, result = self.search_clf_cbf_given_lagrangian(
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
            return V, b, rho, result

        assert scale_max >= scale_min
        assert scale_tol > 0
        V, b, rho, result = search(scale_max)
        if result.is_success():
            print(f"binary_search_clf_cbf: scale={scale_max} is feasible.")
            assert b is not None
            return V, b, rho

        V_success, b_success, rho_success, result = search(scale_min)
        assert (
            result.is_success()
        ), f"binary_search_clf_cbf: scale_min={scale_min} is not feasible."
        assert b_success is not None

        while scale_max - scale_min > scale_tol:
            scale = (scale_max + scale_min) / 2
            V, b, rho, result = search(scale)
            if result.is_success():
                print(f"binary_search_clf_cbf: scale={scale} is feasible.")
                scale_min = scale
                V_success = V
                assert b is not None
                b_success = b
                rho_success = rho
            else:
                print(f"binary_search_clf_cbf: scale={scale} is not feasible.")
                scale_max = scale

        return V_success, b_success, rho_success

    def in_compatible_region(
        self,
        V: Optional[sym.Polynomial],
        b: np.ndarray,
        rho: Optional[float],
        x_samples: np.ndarray,
    ) -> np.ndarray:
        """
        Returns if x_samples[i] is in the compatible region
        {x | V(x) <= rho, b(x) >= 0}.

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
            assert rho is not None
            in_V = V.EvaluateIndeterminates(self.x, x_samples.T) <= rho
            return np.logical_and(in_b, in_V)
        else:
            return in_b

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
        rho: Optional[float],
        barrier_eps: Optional[np.ndarray],
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
        -1 - s₀(x, y)ᵀ Λ(x)ᵀy - s₁(x, y)(ξ(x)ᵀy+1) - s₂(x, y)ᵀy - s₃(x, y)(ρ − V) - s₄(x, y)ᵀ(b(x)+ε) is sos          (3)
        s₂(x, y), s₃(x, y), s₄(x, y) are all sos.

        To certify the emptiness of the set in (2), we can use the sufficient condition
        -1 - s₀(x, y)ᵀ Λ(x)ᵀy² - s₁(x, y)(ξ(x)ᵀy²+1) - s₃(x, y)(ρ − V) - s₄(x, y)ᵀ(b(x)+ε) is sos                     (4)
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

        # Compute s₃(x, y)(ρ − V)
        if rho is not None and self.with_clf:
            assert V is not None
            assert lagrangians.rho_minus_V is not None
            poly -= lagrangians.rho_minus_V * (rho * poly_one - V)

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
    ) -> Tuple[
        solvers.MathematicalProgram,
        Optional[sym.Polynomial],
        np.ndarray,
        Optional[sym.Variable],
    ]:
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
            clf_monomials = sym.MonomialBasis(self.x, int(np.floor(clf_degree / 2)))
            if np.all(x_equilibrium == 0):
                # If the equilibrium state x* = 0, then we know that V(x*)=0
                # and V(x) > 0 forall x != x*. This means that the linear and
                # constant coefficient of V is zero. Hence we remove "1" from
                # clf_monomials.
                clf_monomials = np.array(
                    [
                        monomial
                        for monomial in clf_monomials
                        if monomial.total_degree() != 0
                    ]
                )
            V, clf_gram = prog.NewSosPolynomial(clf_monomials)
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
        rho = None if V is None else prog.NewContinuousVariables(1, "rho")[0]
        if rho is not None:
            prog.AddBoundingBoxConstraint(0, np.inf, rho)

        self._add_compatibility(
            prog=prog,
            V=V,
            b=b,
            kappa_V=kappa_V,
            kappa_b=kappa_b,
            lagrangians=compatible_lagrangians,
            rho=rho,
            barrier_eps=barrier_eps,
        )

        return (prog, V, b, rho)

    def _find_max_inner_ellipsoid(
        self,
        V: Optional[sym.Polynomial],
        b: np.ndarray,
        rho: Optional[float],
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
          x_inner_init: The initial guess on a point inside V(x) <= rho and
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
            assert rho is not None
            V_contain_lagrangian.add_constraint(
                prog,
                inner_ineq_poly=np.array([ellipsoid]),
                inner_eq_poly=self.state_eq_constraints,
                outer_poly=V - rho,
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

        # Make sure x_inner_init is inside V(x) <= rho and b(x) >= 0.
        env_inner_init = {self.x[i]: x_inner_init[i] for i in range(self.nx)}
        if V is not None:
            assert V.Evaluate(env_inner_init) <= rho
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
        rho: sym.Variable,
        S_ellipsoid_inner: np.ndarray,
        b_ellipsoid_inner: np.ndarray,
        c_ellipsoid_inner: float,
    ):
        """
        Add the constraint that the ellipsoid is contained within the
        compatible region {x | V(x) <= rho, b(x) >= 0}.
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
            assert rho is not None
            ellipsoid_in_V_lagrangian.add_constraint(
                prog,
                inner_ineq_poly=np.array([ellipsoid_poly]),
                inner_eq_poly=self.state_eq_constraints,
                outer_poly=V - sym.Polynomial({sym.Monomial(): sym.Expression(rho)}),
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
