"""
Certify the compatible CLF/CBF with or without input limits.
This uses the nonlinear dynamics without the state equation constraints from the
trigonometric polynomials.
"""

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_cbf import clf_cbf
from examples.nonlinear_toy import toy_system


def main(use_y_squared: bool, with_u_bound: bool):
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = toy_system.affine_dynamics(x)
    use_y_squared = True
    if with_u_bound:
        Au = np.array([[1], [-1.0]])
        bu = np.array([20, 20])
    else:
        Au = None
        bu = None
    exclude_sets = [clf_cbf.ExcludeSet(np.array([sym.Polynomial(x[0] + 10)]))]
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        exclude_sets=exclude_sets,
        within_set=None,
        Au=Au,
        bu=bu,
        num_cbf=1,
        with_clf=True,
        use_y_squared=use_y_squared,
    )
    V_init = sym.Polynomial(x[0] ** 2 + x[1] ** 2) / 0.01
    h_init = np.array([sym.Polynomial(0.001 - x[0] ** 2 - x[1] ** 2)])
    kappa_V = 1e-3
    kappa_h = np.array([kappa_V])

    lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[clf_cbf.XYDegree(x=3, y=0)],
        xi_y=clf_cbf.XYDegree(x=2, y=0),
        y=(
            None
            if use_y_squared
            else [clf_cbf.XYDegree(x=4, y=0) for _ in range(compatible.y.size)]
        ),
        rho_minus_V=clf_cbf.XYDegree(x=2, y=2),
        h_plus_eps=[clf_cbf.XYDegree(x=2, y=2)],
        state_eq_constraints=None,
    )
    barrier_eps = np.array([0.0001])
    (
        compatible_prog,
        compatible_lagrangians,
    ) = compatible.construct_search_compatible_lagrangians(
        V_init, h_init, kappa_V, kappa_h, lagrangian_degrees, barrier_eps
    )
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 0)
    compatible_result = solvers.Solve(compatible_prog, None, solver_options)
    assert compatible_result.is_success()

    safety_sets_lagrangian_degrees = clf_cbf.SafetySetLagrangianDegrees(
        exclude=[
            clf_cbf.ExcludeRegionLagrangianDegrees(
                cbf=[0], unsafe_region=[0], state_eq_constraints=None
            )
        ],
        within=[],
    )

    safety_sets_lagrangians = compatible.certify_cbf_safety_set(
        h=h_init,
        lagrangian_degrees=safety_sets_lagrangian_degrees,
        solver_options=None,
    )
    assert safety_sets_lagrangians is not None


if __name__ == "__main__":
    main(use_y_squared=True, with_u_bound=False)
    main(use_y_squared=False, with_u_bound=False)
    main(use_y_squared=True, with_u_bound=True)
    main(use_y_squared=False, with_u_bound=True)
