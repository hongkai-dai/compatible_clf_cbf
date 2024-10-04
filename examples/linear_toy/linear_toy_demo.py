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
    h: np.ndarray,
    kappa_V: float,
    kappa_h: np.ndarray,
) -> clf_cbf.CompatibleLagrangians:
    y_size = dut.y.size

    # Search for the
    lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[clf_cbf.XYDegree(x=2, y=0) for _ in range(dut.nu)],
        xi_y=clf_cbf.XYDegree(x=2, y=0),
        y=(
            None
            if dut.use_y_squared
            else [clf_cbf.XYDegree(x=2, y=0) for _ in range(y_size)]
        ),
        y_cross=None,
        rho_minus_V=None,
        h_plus_eps=[clf_cbf.XYDegree(x=2, y=0)],
        state_eq_constraints=None,
    )
    prog, lagrangians = dut.construct_search_compatible_lagrangians(
        V, h, kappa_V, kappa_h, lagrangian_degrees, barrier_eps=None, local_clf=False
    )

    result = pydrake.solvers.Solve(prog)
    assert result.is_success()
    lagrangians_result = lagrangians.get_result(result, coefficient_tol=1e-8)
    return lagrangians_result


def search_barrier_safe_lagrangians(
    dut: clf_cbf.CompatibleClfCbf, h: np.ndarray
) -> List[clf_cbf.SafetySetLagrangians]:
    lagrangian_degrees = [
        clf_cbf.SafetySetLagrangianDegrees(
            exclude=[
                clf_cbf.ExcludeRegionLagrangianDegrees(
                    cbf=[2], unsafe_region=[2], state_eq_constraints=None
                )
            ],
            within=[],
        )
    ]
    lagrangians = dut.certify_cbf_safety_set(h, lagrangian_degrees[0])
    assert lagrangians is not None
    return [lagrangians]


def search_lagrangians(
    dut: clf_cbf.CompatibleClfCbf,
    V: sym.Polynomial,
    h: np.ndarray,
    kappa_V: float,
    kappa_h: np.ndarray,
) -> Tuple[clf_cbf.CompatibleLagrangians, List[clf_cbf.SafetySetLagrangians]]:
    compatible_lagrangians = search_compatible_lagrangians(dut, V, h, kappa_V, kappa_h)
    barrier_safe_lagrangians = search_barrier_safe_lagrangians(dut, h)
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

    exclude_sets = [
        clf_cbf.ExcludeSet(
            np.array([1.1 * alpha - sym.Polynomial(x.dot(S_lqr @ x))]),
        )
    ]

    dut = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        exclude_sets=exclude_sets,
        within_set=None,
        Au=None,
        bu=None,
        num_cbf=1,
        with_clf=True,
        use_y_squared=use_y_squared,
    )

    V = sym.Polynomial(x.dot(S_lqr @ x))
    h = np.array([alpha - V])
    kappa_V = 0.001
    kappa_h = np.array([0.001])

    search_lagrangians(dut, V, h, kappa_V, kappa_h)


def main():
    search(use_y_squared=True)
    search(use_y_squared=False)


if __name__ == "__main__":
    main()
