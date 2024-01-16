"""
We search for compatible CLF and CBF for a 2D linear system.
"""
from typing import List, Tuple

import numpy as np

import pydrake.systems.controllers
import pydrake.solvers
import pydrake.symbolic as sym

from compatible_clf_cbf import clf_cbf


def search_compatible_lagrangians(
    dut: clf_cbf.CompatibleClfCbf,
    V: sym.Polynomial,
    b: np.ndarray,
    kappa_V: float,
    kappa_b: np.ndarray,
) -> clf_cbf.CompatibleLagrangians:
    y_size = dut.y.size

    # Search for the
    lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[
            clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0) for _ in range(dut.nx)
        ],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0),
        y=None
        if dut.use_y_squared
        else [
            clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0) for _ in range(y_size)
        ],
        rho_minus_V=None,
        b_plus_eps=None,
        state_eq_constraints=None,
    )
    prog, lagrangians = dut.construct_search_compatible_lagrangians(
        V, b, kappa_V, kappa_b, lagrangian_degrees, rho=None, barrier_eps=None
    )

    result = pydrake.solvers.Solve(prog)
    assert result.is_success()
    lagrangians_result = lagrangians.get_result(result, coefficient_tol=1e-8)
    return lagrangians_result


def search_barrier_safe_lagrangians(
    dut: clf_cbf.CompatibleClfCbf, b: np.ndarray
) -> List[clf_cbf.UnsafeRegionLagrangians]:
    lagrangians = dut.certify_cbf_unsafe_region(0, b[0], 2, [2])
    return [lagrangians]


def search_lagrangians(
    dut: clf_cbf.CompatibleClfCbf,
    V: sym.Polynomial,
    b: np.ndarray,
    kappa_V: float,
    kappa_b: np.ndarray,
) -> Tuple[clf_cbf.CompatibleLagrangians, List[np.ndarray]]:
    compatible_lagrangians = search_compatible_lagrangians(dut, V, b, kappa_V, kappa_b)
    barrier_safe_lagrangians = search_barrier_safe_lagrangians(dut, b)
    return (compatible_lagrangians, barrier_safe_lagrangians)


def search(use_y_squared: bool):
    A = np.array([[1, 2], [-2, 3.0]])
    B = np.array([[1, 0], [0, 1.0]])

    # First compute the LQR controller through Riccati equation.
    Q = np.eye(2)
    R = np.eye(2)
    K_lqr, S_lqr = pydrake.systems.controllers.LinearQuadraticRegulator(A, B, Q, R)

    x = sym.MakeVectorContinuousVariable(2, "x")

    Ax = A @ x
    f = np.array([sym.Polynomial(Ax[i]) for i in range(2)])
    g = np.empty(B.shape, dtype=object)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            g[i, j] = sym.Polynomial(B[i, j])

    # Use an arbitrary unsafe region
    alpha = 0.5
    unsafe_regions = [np.array([1.1 * alpha - sym.Polynomial(x.dot(S_lqr @ x))])]

    dut = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        unsafe_regions=unsafe_regions,
        Au=None,
        bu=None,
        with_clf=True,
        use_y_squared=use_y_squared,
    )

    V = sym.Polynomial(x.dot(S_lqr @ x))
    b = np.array([alpha - V])
    kappa_V = 0.001
    kappa_b = np.array([0.001])

    search_lagrangians(dut, V, b, kappa_V, kappa_b)


def main():
    search(use_y_squared=True)
    search(use_y_squared=False)


if __name__ == "__main__":
    main()
