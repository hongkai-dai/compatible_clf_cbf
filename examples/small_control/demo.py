"""
Consider a 1-D system ẋ = x + x²u.
The previous approach fails to verify that V = x² and h = 1-x² are compatible.
Our approach can.
"""

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_cbf import clf_cbf


def main():
    x = np.array([sym.Variable("x")])
    f = np.array([sym.Polynomial(x[0])])
    g = np.array([[sym.Polynomial(x[0] ** 2)]])
    exclude_sets = [clf_cbf.ExcludeSet(np.array([sym.Polynomial(x[0] + 10)]))]
    use_y_squared = True
    compatible = clf_cbf.CompatibleClfCbf(
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
    V = sym.Polynomial(x[0] ** 2)
    h = np.array([1 - V])
    kappa_V = 1
    kappa_h = np.array([1])

    lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[clf_cbf.XYDegree(x=1, y=2)],
        xi_y=clf_cbf.XYDegree(x=0, y=2),
        y=(
            None
            if use_y_squared
            else [clf_cbf.XYDegree(x=6, y=0) for _ in range(compatible.y.size)]
        ),
        y_cross=None,
        rho_minus_V=None,
        h_plus_eps=None,
        lower_lie_derivative=None,
        state_eq_constraints=None,
    )
    barrier_eps = None

    compatible_prog, compatible_lagrangians = (
        compatible.construct_search_compatible_lagrangians(
            V,
            h,
            kappa_V,
            kappa_h,
            lagrangian_degrees,
            barrier_eps,
            local_clf=False,
        )
    )

    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 1)
    compatible_result = solvers.Solve(compatible_prog, None, solver_options)
    compatible_lagrangians_result = compatible_lagrangians.get_result(
        compatible_result, coefficient_tol=1e-10
    )
    assert compatible_result.is_success()
    print(f"xi_y lagrangians\n{compatible_lagrangians_result.xi_y}")
    print(f"lambda_y lagrangians\n{compatible_lagrangians_result.lambda_y}")


if __name__ == "__main__":
    main()
