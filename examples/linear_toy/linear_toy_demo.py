"""
We search for compatible CLF and CBF for a 2D linear system.
"""
import compatible_clf_cbf.clf_cbf as clf_cbf
import numpy as np

import pydrake.systems.controllers
import pydrake.solvers
import pydrake.symbolic as sym


def search(use_y_squared: bool):
    A = np.array([[1, 2], [-2, 3.0]])
    B = np.array([[1, 0], [0, 1.0]])

    # First compute the LQR controller through Riccati equation.
    Q = np.eye(2)
    R = np.eye(2)
    K_lqr, S_lqr = pydrake.systems.controllers.LinearQuadraticRegulator(A, B, Q, R)

    prog = pydrake.solvers.MathematicalProgram()
    x = prog.NewIndeterminates(2, "x")

    Ax = A @ x
    f = np.array([sym.Polynomial(Ax[i]) for i in range(2)])
    g = np.empty(B.shape, dtype=object)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            g[i, j] = sym.Polynomial(B[i, j])

    # Use an arbitrary unsafe region
    alpha = 0.5
    unsafe_regions = [np.array([0.9 * alpha - sym.Polynomial(x.dot(S_lqr @ x))])]

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
    prog.AddIndeterminates(dut.y)

    V = sym.Polynomial(x.dot(S_lqr @ x))
    b = np.array([alpha - V])
    kappa_V = 0.001
    kappa_b = np.array([0.001])
    y_size = dut.y.size

    lagrangians = clf_cbf.CompatibleLagrangians.reserve(
        nu=2,
        use_y_squared=use_y_squared,
        y_size=y_size,
        with_rho_minus_V=False,
        b_plus_eps_size=None,
    )
    lagrangians.lambda_y[0] = prog.NewFreePolynomial(dut.xy_set, deg=4)
    lagrangians.lambda_y[1] = prog.NewFreePolynomial(dut.xy_set, deg=4)
    lagrangians.xi_y = prog.NewFreePolynomial(dut.xy_set, deg=4)
    if not use_y_squared:
        for i in range(y_size):
            lagrangians.y[i] = prog.NewSosPolynomial(dut.xy_set, degree=2)[0]

    dut._add_compatibility(
        prog=prog,
        V=V,
        b=b,
        kappa_V=kappa_V,
        kappa_b=kappa_b,
        lagrangians=lagrangians,
        rho=None,
        barrier_eps=None,
    )

    result = pydrake.solvers.Solve(prog)
    assert result.is_success()
    lagrangians_result = lagrangians.get_result(result, coefficient_tol=1e-8)
    # The Lagrangians are not 0
    assert not lagrangians_result.xi_y.EqualTo(sym.Polynomial())


def main():
    search(use_y_squared=True)
    search(use_y_squared=False)


if __name__ == "__main__":
    main()
