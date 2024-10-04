"""
Synthesize the compatible CLF/CBF with or without input limits.
This uses the nonlinear dynamics without the state equation constraints from the
trigonometric polynomials.
"""

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_cbf import clf_cbf
import compatible_clf_cbf.utils
from examples.nonlinear_toy import toy_system


def main(with_u_bound: bool):
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = toy_system.affine_dynamics(x)
    if with_u_bound:
        Au = np.array([[1], [-1.0]])
        bu = np.array([20, 20])
    else:
        Au = None
        bu = None
    exclude_sets = [clf_cbf.ExcludeSet(np.array([sym.Polynomial(x[0] + 5)]))]
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
        use_y_squared=True,
    )
    V_init = sym.Polynomial(x[0] ** 2 + x[1] ** 2) / 0.01
    h_init = np.array([sym.Polynomial(0.01 - x[0] ** 2 - x[1] ** 2)])
    kappa_V = 1e-3
    kappa_h = np.array([kappa_V])
    barrier_eps = np.array([0.0001])

    compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[clf_cbf.XYDegree(x=3, y=0)],
        xi_y=clf_cbf.XYDegree(x=2, y=0),
        y=None,
        y_cross=None,
        rho_minus_V=clf_cbf.XYDegree(x=2, y=2),
        h_plus_eps=[clf_cbf.XYDegree(x=2, y=2)],
        state_eq_constraints=None,
    )
    safety_sets_lagrangian_degrees = clf_cbf.SafetySetLagrangianDegrees(
        exclude=[
            clf_cbf.ExcludeRegionLagrangianDegrees(
                cbf=[0], unsafe_region=[0], state_eq_constraints=None
            )
        ],
        within=[],
    )

    x_equilibrium = np.array([0.0, 0.0])

    clf_degree = 2
    cbf_degrees = [2]
    max_iter = 4

    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=np.array([[1, 1], [-1, 1]]),
        anchor_states=np.array([[0, 0]]),
        h_anchor_bounds=[(np.array([0]), np.array([0.1]))],
        weight_V=1.0,
        weight_h=np.array([1.0]),
    )

    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 0)
    solver_options.SetOption(solvers.ClarabelSolver.id(), "max_iter", 10000)

    V, h = compatible.bilinear_alternation(
        V_init,
        h_init,
        compatible_lagrangian_degrees,
        safety_sets_lagrangian_degrees,
        kappa_V,
        kappa_h,
        barrier_eps,
        x_equilibrium,
        clf_degree,
        cbf_degrees,
        max_iter,
        inner_ellipsoid_options=None,
        binary_search_scale_options=None,
        compatible_states_options=compatible_states_options,
        solver_options=solver_options,
        backoff_scale=compatible_clf_cbf.utils.BackoffScale(rel=0.02, abs=None),
    )


if __name__ == "__main__":
    main(with_u_bound=False)
