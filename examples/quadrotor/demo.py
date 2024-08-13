"""
Certify the compatible CLF/CBF with input limits.
This uses the polynomial dynamics of the quadrotor (with quaternion as part of
the state), and equality constraint to enforce the unit-length constraint on the
quaternion.
"""

import os

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

import compatible_clf_cbf.clf as clf
from compatible_clf_cbf import clf_cbf
from examples.quadrotor.plant import QuadrotorPolyPlant
import examples.quadrotor.demo_clf
from compatible_clf_cbf.utils import BackoffScale


def main(use_y_squared: bool, with_u_bound: bool):
    x = sym.MakeVectorContinuousVariable(13, "x")
    quadrotor = QuadrotorPolyPlant()
    f, g = quadrotor.affine_dynamics(x)

    if with_u_bound:
        Au = np.concatenate((np.eye(4), -np.eye(4)), axis=0)
        u_bound = quadrotor.m * quadrotor.g
        bu = np.concatenate((np.full((4,), u_bound), np.zeros((4,))))
    else:
        Au, bu = None, None

    # Ground as the unsafe region.
    unsafe_regions = [np.array([sym.Polynomial(x[6] + 0.5)])]
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

    load_V_init: bool = True
    x_set = sym.Variables(x)
    V_degree = 2
    b_degrees = [2]
    if load_V_init:
        data = clf.load_clf(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../data/quadrotor_V_init.pkl",
            ),
            x_set,
        )
        V_init = 3 * data["V"].RemoveTermsWithSmallCoefficients(1e-6)
    else:
        V_init = examples.quadrotor.demo_clf.find_trig_regional_clf(
            V_degree=2, x=x, save_pickle_filename="quadrotor_V_init1.pkl"
        )

    b_init = np.array([1 - V_init])
    kappa_V = 0
    kappa_b = np.array([kappa_V])

    compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[
            clf_cbf.CompatibleLagrangianDegrees.Degree(x=1, y=0 if use_y_squared else 1)
            for _ in range(4)
        ],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(
            x=1, y=0 if use_y_squared else 1
        ),
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
        state_eq_constraints=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2)],
    )
    unsafe_regions_lagrangian_degrees = [
        clf_cbf.UnsafeRegionLagrangianDegrees(
            cbf=0, unsafe_region=[0], state_eq_constraints=[0]
        )
    ]
    barrier_eps = np.array([0.000])
    x_equilibrium = np.zeros((13,))
    candidate_compatible_states = np.zeros((6, 13))
    candidate_compatible_states[1, 6] = -0.3
    candidate_compatible_states[2, 5] = -0.3
    candidate_compatible_states[2, 6] = -0.3
    candidate_compatible_states[3, 5] = 0.3
    candidate_compatible_states[3, 6] = -0.3
    candidate_compatible_states[4, 4] = 0.3
    candidate_compatible_states[4, 6] = -0.3
    candidate_compatible_states[5, 4] = -0.3
    candidate_compatible_states[5, 6] = -0.3
    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=candidate_compatible_states,
        anchor_states=np.zeros((1, 13)),
        b_anchor_bounds=[(np.array([0.0]), np.array([1.0]))],
        weight_V=1,
        weight_b=np.array([1]),
    )
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)

    max_iter = 1
    backoff_scales = [
        BackoffScale(rel=None, abs=0.2),
    ]
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
        max_iter=max_iter,
        solver_options=solver_options,
        lagrangian_coefficient_tol=None,
        compatible_states_options=compatible_states_options,
        backoff_scales=backoff_scales,
        lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
    )
    pickle_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/quadrotor_clf_cbf.pkl"
    )
    clf_cbf.save_clf_cbf(V, b, x_set, kappa_V, kappa_b, pickle_path)
    compatible_prog, compatible_lagrangians = (
        compatible.construct_search_compatible_lagrangians(
            V,
            b,
            kappa_V=1e-3,
            kappa_b=np.array([1e-3]),
            lagrangian_degrees=compatible_lagrangian_degrees,
            barrier_eps=barrier_eps,
            local_clf=True,
        )
    )
    compatible_result = solvers.Solve(compatible_prog, None, solver_options)
    assert compatible_result.is_success()


if __name__ == "__main__":
    main(use_y_squared=True, with_u_bound=False)
