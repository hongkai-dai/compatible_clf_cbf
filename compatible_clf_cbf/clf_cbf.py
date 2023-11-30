from dataclasses import dataclass
from typing import List, Optional, Tuple
from typing_extensions import Self

import numpy as np

import pydrake.symbolic as sym
import pydrake.solvers as solvers

from compatible_clf_cbf.utils import check_array_of_polynomials, get_polynomial_result


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

    @classmethod
    def reserve(
        cls,
        nu: int,
        use_y_squared: bool,
        y_size,
        with_rho_minus_V: bool,
        b_plus_eps_size: Optional[int],
    ) -> Self:
        """
        Reserve the Lagrangian polynomials. Note that the polynomials are
        initialized to 0, you should properly set their values.

        Args:
          nu: The dimension of control u.
          use_y_squared: Check CompatibleClfCbf documentation.
          y_size: The size of the indeterminates y.
          with_rho_minus_V: Whether the psatz condition considers ρ - V.
          b_plus_eps_size: The size of b(x)+ε. If set to None, then we don't consider
          b(x)+ε in the psatz.
        """
        return CompatibleLagrangians(
            lambda_y=np.array([sym.Polynomial() for _ in range(nu)]),
            xi_y=sym.Polynomial(),
            y=None
            if use_y_squared
            else np.array([sym.Polynomial() for _ in range(y_size)]),
            rho_minus_V=sym.Polynomial() if with_rho_minus_V else None,
            b_plus_eps=None
            if b_plus_eps_size is None
            else np.array([sym.Polynomial() for _ in range(b_plus_eps_size)]),
        )

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
        return CompatibleLagrangians(
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            y=y_result,
            rho_minus_V=rho_minus_V_result,
            b_plus_eps=b_plus_eps_result,
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
        return CompatibleLagrangians(
            lambda_y=lambda_y,
            xi_y=xi_y,
            y=y_lagrangian,
            rho_minus_V=rho_minus_V,
            b_plus_eps=b_plus_eps,
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

    def get_result(self, result: solvers.MathematicalProgramResult) -> Self:
        return UnsafeRegionLagrangians(
            cbf=result.GetSolution(self.cbf),
            unsafe_region=np.array(
                [result.GetSolution(phi) for phi in self.unsafe_region]
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

    def certify_cbf_unsafe_region(
        self,
        unsafe_region_index: int,
        cbf: sym.Polynomial,
        cbf_lagrangian_degree: int,
        unsafe_region_lagrangian_degrees: List[int],
        solver_options: Optional[solvers.SolverOptions] = None,
    ) -> UnsafeRegionLagrangians:
        """
        Certifies that the 0-superlevel set {x | bᵢ(x) >= 0} does not intersect
        with the unsafe region self.unsafe_regions[unsafe_region_index].

        If we denote the unsafe region as {x | q(x) <= 0}, then we impose the constraint

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
        lagrangians = UnsafeRegionLagrangians(cbf_lagrangian, unsafe_lagrangians)
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
            assert lagrangians.b_plus_eps is not None
            poly -= lagrangians.b_plus_eps.dot(barrier_eps + b)

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
          poly: poly is the polynomial -1-ϕᵢ,₀(x)bᵢ(x) + ∑ⱼϕᵢ,ⱼ(x)pⱼ(x)
        """
        assert lagrangians.unsafe_region.size == len(
            self.unsafe_regions[unsafe_region_index]
        )
        poly = -(1 + lagrangians.cbf) * b + lagrangians.unsafe_region.dot(
            self.unsafe_regions[unsafe_region_index]
        )
        prog.AddSosConstraint(poly)
        return poly
