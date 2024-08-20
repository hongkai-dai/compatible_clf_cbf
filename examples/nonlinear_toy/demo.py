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
    safety_sets = [
        clf_cbf.SafetySet(exclude=np.array([sym.Polynomial(x[0] + 10)]), within=None)
    ]
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        safety_sets=safety_sets,
        Au=Au,
        bu=bu,
        with_clf=True,
        use_y_squared=use_y_squared,
    )
    V_init = sym.Polynomial(x[0] ** 2 + x[1] ** 2) / 0.01
    b_init = np.array([sym.Polynomial(0.001 - x[0] ** 2 - x[1] ** 2)])
    kappa_V = 1e-3
    kappa_b = np.array([kappa_V])

    lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=3, y=0)],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0),
        y=(
            None
            if use_y_squared
            else [
                clf_cbf.CompatibleLagrangianDegrees.Degree(x=4, y=0)
                for _ in range(compatible.y.size)
            ]
        ),
        rho_minus_V=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2),
        b_plus_eps=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2)],
        state_eq_constraints=None,
    )
    barrier_eps = np.array([0.0001])
    (
        compatible_prog,
        compatible_lagrangians,
    ) = compatible.construct_search_compatible_lagrangians(
        V_init, b_init, kappa_V, kappa_b, lagrangian_degrees, barrier_eps
    )
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 0)
    compatible_result = solvers.Solve(compatible_prog, None, solver_options)
    assert compatible_result.is_success()

    safety_sets_lagrangian_degrees = [
        clf_cbf.SafetySetLagrangianDegrees(
            exclude=clf_cbf.ExcludeRegionLagrangianDegrees(
                cbf=0, unsafe_region=[0], state_eq_constraints=None
            ),
            within=None,
        )
    ]

    safety_sets_lagrangians = [
        compatible.certify_cbf_safety_set(
            safety_set_index=0,
            cbf=b_init[0],
            lagrangian_degrees=safety_sets_lagrangian_degrees[0],
            solver_options=None,
        )
    ]
    assert safety_sets_lagrangians[0] is not None


if __name__ == "__main__":
    main(use_y_squared=True, with_u_bound=False)
    main(use_y_squared=False, with_u_bound=False)
    main(use_y_squared=True, with_u_bound=True)
    main(use_y_squared=False, with_u_bound=True)
