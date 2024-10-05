"""
Certify the compatible CLF/CBF using taylor expansion of the 12-state quadrotor
dynamics.
"""

import itertools
import os

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

import compatible_clf_cbf.clf_cbf as clf_cbf
import compatible_clf_cbf.clf as clf
from compatible_clf_cbf.utils import BackoffScale
from examples.quadrotor.plant import QuadrotorPlant


def search(use_y_squared: bool, with_u_bound: bool, use_v_rep: bool):
    x = sym.MakeVectorContinuousVariable(12, "x")
    quadrotor = QuadrotorPlant()
    f, g = quadrotor.affine_dynamics_taylor(x, np.zeros((12,)), f_degree=3, g_degree=2)

    if with_u_bound:
        u_bound = quadrotor.m * quadrotor.g
        if use_v_rep:
            u_vertices = np.array(list(itertools.product([0, u_bound], repeat=4)))
            u_extreme_rays = None
            Au = None
            bu = None
        else:
            Au = np.concatenate((np.eye(4), -np.eye(4)), axis=0)
            bu = np.concatenate((np.full((4,), u_bound), np.zeros((4,))))
            u_vertices = None
            u_extreme_rays = None
    else:
        u_vertices = None
        u_extreme_rays = None
        Au, bu = None, None

    exclude_sets = [clf_cbf.ExcludeSet(np.array([sym.Polynomial(x[2] + 2.5)]))]

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
        state_eq_constraints=None,
    )

    x_set = sym.Variables(x)
    V_init = clf.load_clf(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/quadrotor_taylor_clf.pkl",
        ),
        x_set,
    )["V"]

    h_init = np.array([1 - V_init])

    if with_u_bound and use_v_rep:
        compatible_lagrangian_degrees = clf_cbf.CompatibleWVrepLagrangianDegrees(
            u_vertices=[clf_cbf.XYDegree(x=2, y=0) for _ in range(u_vertices.shape[0])],
            u_extreme_rays=None,
            xi_y=None,
            y=(
                None
                if use_y_squared
                else [clf_cbf.XYDegree(x=6, y=0) for _ in range(compatible.y.size)]
            ),
            y_cross=(
                None
                if use_y_squared
                else [
                    clf_cbf.XYDegree(x=4, y=0)
                    for _ in range(compatible.y_cross_poly.size)
                ]
            ),
            rho_minus_V=clf_cbf.XYDegree(x=4, y=2, homogeneous_y=True),
            h_plus_eps=[clf_cbf.XYDegree(x=4, y=2, homogeneous_y=True)],
            state_eq_constraints=None,
        )
    else:
        compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
            lambda_y=[clf_cbf.XYDegree(x=2, y=0) for _ in range(4)],
            xi_y=clf_cbf.XYDegree(x=1, y=0),
            y=(
                None
                if use_y_squared
                else [clf_cbf.XYDegree(x=4, y=0) for _ in range(compatible.y.size)]
            ),
            y_cross=(
                None
                if use_y_squared
                else [
                    clf_cbf.XYDegree(x=4, y=0)
                    for _ in range(compatible.y_cross_poly.size)
                ]
            ),
            rho_minus_V=clf_cbf.XYDegree(x=2, y=2, homogeneous_y=True),
            h_plus_eps=[clf_cbf.XYDegree(x=2, y=2, homogeneous_y=True)],
            state_eq_constraints=None,
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
    x_equilibrium = np.zeros((12,))

    candidate_compatible_states = np.zeros((4, 12))
    candidate_compatible_states[0, :3] = np.array([-1.5, 0, 0])
    candidate_compatible_states[1, :3] = np.array([1.5, 0, 0])
    candidate_compatible_states[2, :3] = np.array([0, 1.5, 0])
    candidate_compatible_states[3, :3] = np.array([0, -1.5, 0])

    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=candidate_compatible_states,
        anchor_states=np.zeros((1, 12)),
        h_anchor_bounds=[(np.array([0.5]), np.array([1.0]))],
        weight_V=1,
        weight_h=np.array([1]),
        V_margin=None,
        h_margins=None,
    )

    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)

    kappa_V = 0.1
    kappa_h = np.array([0.1])
    V_degree = 2
    h_degrees = [2]
    backoff_scale = BackoffScale(rel=None, abs=0.001)
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
        max_iter=5,
        solver_options=solver_options,
        lagrangian_coefficient_tol=None,
        compatible_states_options=compatible_states_options,
        backoff_scale=backoff_scale,
        lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    )


def main():
    # search(use_y_squared=True, with_u_bound=False, use_v_rep=False)
    # Using H-rep will cause out-of-memory issue.
    # search(use_y_squared=True, with_u_bound=True, use_v_rep=False)
    search(use_y_squared=True, with_u_bound=True, use_v_rep=True)


if __name__ == "__main__":
    main()
