"""
Synthesize the compatible CLF/CBF with or without input limits.
This uses the nonlinear dynamics without the state equation constraints from the
trigonometric polynomials.
"""

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_cbf import clf_cbf
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
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        unsafe_regions=[np.array([sym.Polynomial(x[0] + 5)])],
        Au=Au,
        bu=bu,
        with_clf=True,
        use_y_squared=True,
    )
    V_init = sym.Polynomial(x[0] ** 2 + x[1] ** 2) / 0.01
    b_init = np.array([sym.Polynomial(0.01 - x[0] ** 2 - x[1] ** 2)])
    kappa_V = 1e-3
    kappa_b = np.array([kappa_V])
    barrier_eps = np.array([0.0001])

    compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=3, y=0)],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0),
        y=None,
        rho_minus_V=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2),
        b_plus_eps=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2)],
        state_eq_constraints=None,
    )
    unsafe_region_lagrangian_degrees = [
        clf_cbf.UnsafeRegionLagrangianDegrees(
            cbf=0, unsafe_region=[0], state_eq_constraints=None
        )
    ]
    x_equilibrium = np.array([0.0, 0.0])

    clf_degree = 2
    cbf_degrees = [2]
    max_iter = 10

    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=np.array([[1, 1], [-1, 1]]),
        anchor_states=np.array([[0, 0]]),
        b_anchor_bounds=[(np.array([0]), np.array([0.1]))],
        weight_V=1.0,
        weight_b=np.array([1.0]),
    )

    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 0)
    solver_options.SetOption(solvers.ClarabelSolver.id(), "max_iter", 10000)

    V, b = compatible.bilinear_alternation(
        V_init,
        b_init,
        compatible_lagrangian_degrees,
        unsafe_region_lagrangian_degrees,
        kappa_V,
        kappa_b,
        barrier_eps,
        x_equilibrium,
        clf_degree,
        cbf_degrees,
        max_iter,
        x_inner=x_equilibrium,
        binary_search_scale_options=None,
        find_inner_ellipsoid_max_iter=0,
        compatible_states_options=compatible_states_options,
        solver_options=solver_options,
        backoff_scale=0.02,
    )


if __name__ == "__main__":
    main(with_u_bound=False)