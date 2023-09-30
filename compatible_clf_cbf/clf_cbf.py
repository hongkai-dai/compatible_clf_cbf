from typing import List, Optional

import numpy as np

import pydrake.symbolic as sym

from compatible_clf_cbf.utils import check_array_of_polynomials


class CompatibleClfCbf:
    """
    Certify and synthesize compatible Control Lyapunov Function (CLF) and
    Control Barrier Functions (CBFs).

    For a continuous-time control affine system
    ẋ = f(x)+g(x)u, u∈𝒰
    A CLF V(x) and a CBF b(x) is compatible if and only if
    ∃ u∈𝒰,      ∂b/∂x*f(x) + ∂b/∂x*g(x)*u ≥ −κ_b*b(x)
            and ∂V/∂x*f(x) + ∂V/∂x*g(x)*u ≤ −κ_V*V(x)
    For simplicity, let's first consider that u is un-constrained, namely 𝒰 is
    the entire space.
    By Farkas lemma, this is equivalent to the following set being empty
    {(x, y) | yᵀ*[∂b/∂x*g(x)] = 0, yᵀ*[-∂b/∂x*f(x)-κ_b*b(x)] = -1, y>=0}
                 [∂V/∂x*g(x)]         [ ∂V/∂x*f(x)+κ_V*V(x)]
    We can then use Positivstellensatz to certify the emptiness of this set.

    The same math applies to multiple CBFs, or when u is constrained within a
    polyhedron.
    """

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
        use_y_squared: bool = True
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

            {(x, y) | y²ᵀ*[∂b/∂x*g(x)] = 0, y²ᵀ*[-∂b/∂x*f(x)-κ_b*b(x)] = -1}
                          [∂V/∂x*g(x)]          [ ∂V/∂x*f(x)+κ_V*V(x)]
            is empty.

          If both Au and bu are None, it means that we don't have input limits.
          They have to be both None or both not None.
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
