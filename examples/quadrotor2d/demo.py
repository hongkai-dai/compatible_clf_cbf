"""
Certify the compatible CLF/CBF with input limits.
This uses the polynomial dynamics of the 2D quadrotor (with sin/cos as part of
the state), and equality constraint to enforce the unit-length constraint on the
cos/sin.
"""

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_cbf import clf_cbf
from examples.quadrotor2d.plant import Quadrotor2dTrigPlant
import examples.quadrotor2d.demo_clf as demo_clf


def main(use_y_squared: bool, with_u_bound: bool):
    x = sym.MakeVectorContinuousVariable(7, "x")
    quadrotor = Quadrotor2dTrigPlant()
    f, g = quadrotor.affine_dynamics(x)

    if with_u_bound:
        Au = np.concatenate((np.eye(2), -np.eye(2)), axis=0)
        u_bound = quadrotor.m * quadrotor.g * 1.5
        bu = np.concatenate((np.full((2,), u_bound), np.zeros((2,))))
    else:
        Au, bu = None, None

    # Ground as the unsafe region.
    unsafe_regions = [np.array([sym.Polynomial(x[1] + 0.5)])]
    state_eq_constraints = quadrotor.equality_constraint(x)
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        unsafe_regions=unsafe_regions,
        Au=Au,
        bu=bu,
        with_clf=True,
        use_y_squared=use_y_squared,
        state_eq_constraints=state_eq_constraints,
    )

    V_degree = 2
    V_init = 10 * demo_clf.find_trig_regional_clf(V_degree, x)
    b_degrees = [2]
    b_init = np.array([1 - V_init])
    kappa_V = 0
    kappa_b = np.array([kappa_V])
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)

    compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[
            clf_cbf.CompatibleLagrangianDegrees.Degree(x=1, y=1) for _ in range(2)
        ],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=1),
        y=(
            None
            if use_y_squared
            else [
                clf_cbf.CompatibleLagrangianDegrees.Degree(x=4, y=1)
                for _ in range(compatible.y.size)
            ]
        ),
        rho_minus_V=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2),
        b_plus_eps=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2)],
        state_eq_constraints=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2)],
    )
    unsafe_regions_lagrangian_degrees = [
        clf_cbf.UnsafeRegionLagrangianDegrees(
            cbf=0, unsafe_region=[0], state_eq_constraints=[0]
        )
    ]
    barrier_eps = np.array([0.000])
    x_equilibrium = np.zeros((7,))
    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, -0.3, 0, 0, 0, 0, 0],
                [0.4, -0.3, 0, 0, 0, 0, 0],
                [-0.4, -0.3, 0, 0, 0, 0, 0],
            ]
        ),
        anchor_states=np.zeros((1, 7)),
        b_anchor_bounds=[(np.array([0.0]), np.array([1]))],
        weight_V=1,
        weight_b=np.array([1]),
    )

    V, b = compatible.bilinear_alternation(
        V_init,
        b_init,
        compatible_lagrangian_degrees,
        unsafe_regions_lagrangian_degrees,
        kappa_V,
        kappa_b,
        barrier_eps,
        x_equilibrium,
        V_degree,
        b_degrees,
        max_iter=5,
        solver_options=solver_options,
        # solver_id = solvers.ClarabelSolver().id(),
        lagrangian_coefficient_tol=None,
        compatible_states_options=compatible_states_options,
        backoff_scale=0.02,
    )


if __name__ == "__main__":
    main(use_y_squared=False, with_u_bound=False)
