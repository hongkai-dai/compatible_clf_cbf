import os

import itertools
import numpy as np
import pydrake.symbolic as sym
import pydrake.solvers as solvers
import pydrake.systems.controllers as controllers

from typing import Tuple, List, Union
from compatible_clf_cbf.utils import BackoffScale
from compatible_clf_cbf import clf_cbf
from examples.adaptive_cruise_control import plant


def get_pkl_file_path() -> Tuple[str, str]:
    filename_synth = "acc_clf_cbf.pkl"
    path_synth = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename_synth
    )
    filename_init = "acc_clf_cbf_init.pkl"
    path_init = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename_init
    )
    return (path_init, path_synth)


def save_init_synth_results(
    V: sym.Polynomial,
    h: np.ndarray,
    V_init: sym.Polynomial,
    h_init: np.ndarray,
    x_set: sym.Variables,
    kappaV: float,
    kappah: Union[List, np.ndarray],
):
    """
    The function that saves the initial and synthesis results.
    """
    paths = get_pkl_file_path()
    # save the initial results:
    clf_cbf.save_clf_cbf(
        V=V_init,
        h=h_init,
        x_set=x_set,
        kappa_V=kappaV,
        kappa_h=kappah,
        pickle_path=paths[0],
    )
    # save the synthesis results:
    clf_cbf.save_clf_cbf(
        V=V, h=h, x_set=x_set, kappa_V=kappaV, kappa_h=kappah, pickle_path=paths[1]
    )


def main(with_u_bound: bool, use_v_rep: bool, save_results: bool):
    # system model:
    x = sym.MakeVectorContinuousVariable(2, "x")
    cruise_control = plant.AdaptiveCruiseControlPlant()
    f, g = cruise_control.affine_dynamics(x)

    # physical input constraints:
    # u ∈ [u_min, u_max]
    u_min = -4
    u_max = 3
    nu = 1
    if with_u_bound and use_v_rep:
        u_vertices = np.array(list(itertools.product([u_min, u_max])))
        u_extreme_rays = None
        Au = None
        bu = None
    elif with_u_bound:
        Au = np.concatenate([np.eye(nu), -np.eye(nu)], axis=0)
        bu = np.array([u_max, -u_min])
        u_vertices = None
        u_extreme_rays = None
    else:
        Au = None
        bu = None
        u_vertices = None
        u_extreme_rays = None

    # compute the V(x) using the LQR cost to go function. This could avoid the issue of
    # non-uniform relative degree.
    (A, B) = cruise_control.linear_dynamics(x)
    Q = np.eye(2)
    R = np.eye(1)
    _, S_lqr = controllers.LinearQuadraticRegulator(A, B, Q, R)

    # specify the V(x), b(x) and unsafe regions:
    # V(x) should be Polynomial
    # h(x) should be an array of polynomials
    # unsafe regions should be put in a list of arrays.
    # Each element of the list is a single constriant for a single cbf
    # but each single constraint could be presented by a group of polynomials.
    # exclude sets and within set defines the unsafe region.
    V_init = sym.Polynomial(np.dot(x, np.dot(S_lqr, x)))
    h_init = np.array([sym.Polynomial(-x[1] + 0.5), sym.Polynomial(x[1] + 0.5)])
    exclude_sets = [
        # clf_cbf.ExcludeSet(
        #     np.array([
        #       sym.Polynomial(-x[1] + (cruise_control.d_max - cruise_control.d_ref))
        #     ])
        # ),
        # clf_cbf.ExcludeSet(
        #     np.array(
        #         [sym.Polynomial(x[1] - (cruise_control.d_min - cruise_control.d_ref))]
        #     )
        # ),
    ]
    within_set = clf_cbf.WithinSet(
        np.array(
            [
                -sym.Polynomial(-x[1] + (cruise_control.d_max - cruise_control.d_ref)),
                -sym.Polynomial(x[1] - (cruise_control.d_min - cruise_control.d_ref)),
            ]
        )
    )

    # specify the cbf relative degrees, alpha and betas:
    # relative degrees are put in a list, the size of the list should be equal to the
    # number of cbfs.
    # alpha is just a float for CLF constraint
    # betas are put in list of list of floats, the outter number of elements should be
    # the same as the number of cbf, the inner number of elements should be the same as
    # relative degrees.
    relative_degrees = [2, 2]
    kappaV = 0.1
    kappah = [[2, 2], [2, 2]]
    barrier_eps = np.array([0.0])

    # create comaptibleCLFCBF object:
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        exclude_sets=exclude_sets,
        within_set=within_set,
        Au=Au,
        bu=bu,
        u_vertices=u_vertices,
        u_extreme_rays=u_extreme_rays,
        num_cbf=2,
        high_order_cbf=True,
        relative_degrees=relative_degrees,
        cbf_states=None,
        with_clf=True,
        use_y_squared=True,
        state_eq_constraints=None,
    )
    # specify lagrangian degrees:
    if with_u_bound and use_v_rep:
        compatible_lagrangian_degrees = clf_cbf.CompatibleWVrepLagrangianDegrees(
            u_vertices=[clf_cbf.XYDegree(x=2, y=0) for _ in range(u_vertices.shape[0])],
            u_extreme_rays=None,
            y=None,
            y_cross=None,
            rho_minus_V=clf_cbf.XYDegree(x=2, y=2),
            h_plus_eps=[clf_cbf.XYDegree(x=2, y=2), clf_cbf.XYDegree(x=2, y=2)],
            lower_lie_derivative=[
                [clf_cbf.XYDegree(x=2, y=2)],
                [clf_cbf.XYDegree(x=2, y=2)],
            ],
            state_eq_constraints=None,
        )
    else:
        compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
            lambda_y=[clf_cbf.XYDegree(x=2, y=0)],
            xi_y=clf_cbf.XYDegree(x=2, y=0),
            y=None,
            y_cross=None,
            rho_minus_V=clf_cbf.XYDegree(x=2, y=2),
            h_plus_eps=[clf_cbf.XYDegree(x=2, y=2), clf_cbf.XYDegree(x=2, y=2)],
            lower_lie_derivative=[
                [clf_cbf.XYDegree(x=2, y=2)],
                [clf_cbf.XYDegree(x=2, y=2)],
            ],
            state_eq_constraints=None,
        )

    safety_lagrangian_degrees = clf_cbf.SafetySetLagrangianDegrees(
        exclude=[
            # clf_cbf.ExcludeRegionLagrangianDegrees(
            #     cbf=[2, 2], unsafe_region=[2], state_eq_constraints=None
            # ),
            # clf_cbf.ExcludeRegionLagrangianDegrees(
            #     cbf=[0, 0], unsafe_region=[0], state_eq_constraints=None
            # ),
        ],
        within=[
            clf_cbf.WithinRegionLagrangianDegrees(
                cbf=[2, 2], safe_region=2, state_eq_constraints=None
            ),
            clf_cbf.WithinRegionLagrangianDegrees(
                cbf=[2, 2], safe_region=2, state_eq_constraints=None
            ),
        ],
    )

    # specify clf and cbf degrees, also the state variables for the cbfs
    max_iter = 15
    clf_degree = 2
    cbf_degrees = [1, 1]

    # In this part, the states to include should be in the state space.
    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=np.array(
            [[0, -1.25], [0, 1.25], [0.5, -1.0], [-1, 1.5], [-1, 1.8], [0.5, -1.5]]
        ),
        anchor_states=np.array([[0, 0]]),
        h_anchor_bounds=[
            (np.array([0]), np.array([1.5])),
            (np.array([0]), np.array([1.5])),
        ],
        weight_V=1.0,
        weight_h=np.array([1.0, 1.0]),
        relative_degrees=relative_degrees,
        weight_lower_lie_derivatives=[np.array([1.0]), np.array([1.0])],
        V_margin=0.1,
        h_margins=np.array([0.0, 0.0]),
    )

    # paramters manual tuning:
    # 1. backoff scale:
    if with_u_bound and (not use_v_rep):
        backoff_scale_list = [BackoffScale(rel=None, abs=0.01)] * max_iter
        backoff_scale_list[1] = BackoffScale(rel=None, abs=0.05)
        backoff_scale_list[2] = BackoffScale(rel=None, abs=0.08)
        backoff_scale_list[3] = BackoffScale(rel=None, abs=0.05)
        backoff_scale_list[13] = BackoffScale(rel=None, abs=0.05)
        backoff_scale_list[14] = BackoffScale(rel=None, abs=0.08)
    if with_u_bound and use_v_rep:
        backoff_scale_list = [BackoffScale(rel=None, abs=0.01)] * max_iter
        backoff_scale_list[1] = BackoffScale(rel=None, abs=0.1)
        backoff_scale_list[10] = BackoffScale(rel=None, abs=0.05)
    else:
        backoff_scale_list = None

    # 2. lagraing coefficient tolerance list:
    lagrangian_coefficient_tol = None

    # start the bilinear alternation for synthesis:
    # synthesis:
    V, h = compatible.bilinear_alternation(
        V_init=V_init,
        h_init=h_init,
        compatible_lagrangian_degrees=compatible_lagrangian_degrees,
        safety_sets_lagrangian_degrees=safety_lagrangian_degrees,
        kappa_V=kappaV,
        kappa_h=kappah,
        barrier_eps=barrier_eps,
        x_equilibrium=np.array([0, 0]),
        clf_degree=clf_degree,
        cbf_degrees=cbf_degrees,
        max_iter=max_iter,
        record_time=True,
        compatible_states_options=compatible_states_options,
        backoff_scale=backoff_scale_list,
        lagrangian_coefficient_tol=lagrangian_coefficient_tol,
        lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    )

    assert V is not None
    assert h is not None

    # save the results if needed:
    if save_results:
        save_init_synth_results(
            V=V,
            h=h,
            V_init=V_init,
            h_init=h_init,
            x_set=sym.Variables(x),
            kappaV=kappaV,
            kappah=kappah,
        )


if __name__ == "__main__":
    main(with_u_bound=True, use_v_rep=True, save_results=True)
    # we can set use_v_rep to True or False to see the difference
    # H-rep and V-rep in synthesis time and verification time.
    # Set save_results to True to save the synthesis results, False otherwise.
