"""
Certify the compatible CLF/CBF with input limits.
This uses the polynomial dynamics of the 2D quadrotor (with sin/cos as part of
the state), and equality constraint to enforce the unit-length constraint on the
cos/sin.
"""

import itertools
import os

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_cbf import clf_cbf
import compatible_clf_cbf.clf as clf
from examples.quadrotor2d.plant import Quadrotor2dTrigPlant
import compatible_clf_cbf.utils


def main(use_y_squared: bool, with_u_bound: bool, use_v_rep: bool):
    x = sym.MakeVectorContinuousVariable(7, "x")
    quadrotor = Quadrotor2dTrigPlant()
    f, g = quadrotor.affine_dynamics(x)

    if with_u_bound:
        u_bound = quadrotor.m * quadrotor.g * 1.5
        if use_v_rep:
            u_vertices = np.array(list(itertools.product([0, u_bound], repeat=2)))
            u_extreme_rays = None
            Au, bu = None, None
        else:
            Au = np.concatenate((np.eye(2), -np.eye(2)), axis=0)
            bu = np.concatenate((np.full((2,), u_bound), np.zeros((2,))))
            u_vertices, u_extreme_rays = None, None
    else:
        Au, bu = None, None
        u_vertices, u_extreme_rays = None, None

    # Ground as the unsafe region.
    exclude_sets = [clf_cbf.ExcludeSet(np.array([sym.Polynomial(x[1] + 2)]))]
    state_eq_constraints = quadrotor.equality_constraint(x)
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        exclude_sets=exclude_sets,
        within_set=None,
        Au=Au,
        bu=bu,
        u_vertices=u_vertices,
        u_extreme_rays=u_extreme_rays,
        num_cbf=1,
        with_clf=True,
        use_y_squared=use_y_squared,
        state_eq_constraints=state_eq_constraints,
    )

    V_degree = 2
    # V_init = 10 * demo_clf.find_trig_regional_clf(V_degree, x)
    clf_data = clf.load_clf(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/quadrotor2d_clf.pkl"
        ),
        compatible.x_set,
    )
    V_init = clf_data["V"]
    h_degrees = [2]
    h_init = np.array([1 - V_init])
    kappa_V = 0.1
    kappa_h = np.array([kappa_V])
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)

    if use_v_rep and with_u_bound:
        compatible_lagrangian_degrees = clf_cbf.CompatibleWVrepLagrangianDegrees(
            u_vertices=[clf_cbf.XYDegree(x=2, y=0) for _ in range(u_vertices.shape[0])],
            u_extreme_rays=None,
            xi_y=None,
            y=(
                None
                if use_y_squared
                else [clf_cbf.XYDegree(x=4, y=2) for _ in range(compatible.y.size)]
            ),
            y_cross=None,
            rho_minus_V=clf_cbf.XYDegree(x=2, y=2),
            h_plus_eps=[clf_cbf.XYDegree(x=2, y=2)],
            state_eq_constraints=[clf_cbf.XYDegree(x=4, y=2)],
        )
    else:
        compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
            lambda_y=[
                clf_cbf.XYDegree(x=1, y=0 if use_y_squared else 1) for _ in range(2)
            ],
            xi_y=clf_cbf.XYDegree(x=2, y=0 if use_y_squared else 1),
            y=(
                None
                if use_y_squared
                else [clf_cbf.XYDegree(x=4, y=0) for _ in range(compatible.y.size)]
            ),
            y_cross=None,
            rho_minus_V=clf_cbf.XYDegree(x=2, y=2),
            h_plus_eps=[clf_cbf.XYDegree(x=2, y=2)],
            state_eq_constraints=[clf_cbf.XYDegree(x=2, y=2)],
        )
    safety_sets_lagrangian_degrees = clf_cbf.SafetySetLagrangianDegrees(
        exclude=[
            clf_cbf.ExcludeRegionLagrangianDegrees(
                cbf=[0], unsafe_region=[0], state_eq_constraints=[0]
            )
        ],
        within=[],
    )

    barrier_eps = np.array([0.000])
    x_equilibrium = np.zeros((7,))
    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=np.array(
            [
                [-3, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0],
                [-1.5, -0.5, 0, 0, 0, 0, 0],
                [1.5, 0.5, 0, 0, 0, 0, 0],
                [0, -1.5, 0, 0, 0, 0, 0],
            ]
        ),
        anchor_states=np.zeros((1, 7)),
        h_anchor_bounds=[(np.array([0.0]), np.array([1]))],
        weight_V=1,
        weight_h=np.array([1]),
        h_margins=np.array([0.02]),
    )

    max_iter = 5
    V, h = compatible.bilinear_alternation(
        V_init,
        h_init,
        compatible_lagrangian_degrees,
        safety_sets_lagrangian_degrees,
        kappa_V,
        kappa_h,
        barrier_eps,
        x_equilibrium,
        V_degree,
        h_degrees,
        max_iter=max_iter,
        solver_options=solver_options,
        # solver_id = solvers.ClarabelSolver().id(),
        lagrangian_coefficient_tol=None,
        compatible_states_options=compatible_states_options,
        backoff_scale=compatible_clf_cbf.utils.BackoffScale(rel=None, abs=0.01),
    )
    x_set = sym.Variables(x)

    filename = (
        "quadrotor2d_clf_cbf"
        + ("_w_y_square" if use_y_squared else "_wo_y_squared")
        + ("_vrep" if use_v_rep else "_hrep")
        + ("_w_u_bound" if with_u_bound else "_wo_u_bound")
        + ".pkl"
    )
    pickle_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    clf_cbf.save_clf_cbf(
        V,
        h,
        x_set,
        kappa_V,
        kappa_h,
        pickle_path,
    )


if __name__ == "__main__":
    # main(use_y_squared=False, with_u_bound=False, use_v_rep=False)
    main(use_y_squared=True, with_u_bound=True, use_v_rep=False)
    # main(use_y_squared=True, with_u_bound=True, use_v_rep=True)
