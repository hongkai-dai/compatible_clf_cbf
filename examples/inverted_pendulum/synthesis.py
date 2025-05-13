import os

import itertools
import numpy as np
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers

from typing import Tuple, List, Union
from compatible_clf_cbf import clf_cbf
from examples.inverted_pendulum import plant


def get_pkl_file_path() -> Tuple[str, str]:
    filename_synth = "inverted_pendulum_clf_cbf.pkl"
    path_synth = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename_synth
    )
    filename_init = "inverted_pendulum_clf_cbf_init.pkl"
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
            pickle_path=paths[0]
        )
    # save the synthesis results:
    clf_cbf.save_clf_cbf(
            V=V,
            h=h,
            x_set=x_set,
            kappa_V=kappaV,
            kappa_h=kappah,
            pickle_path=paths[1]
        )


def main(with_u_bound: bool, use_v_rep: bool, save_results: bool):
    # system model:
    x = sym.MakeVectorContinuousVariable(3, "x")
    pendulum = plant.InvertedPendulumPlant()
    f, g = pendulum.trig_poly_affine_dynamics(x)
    state_eq_constr = np.array([pendulum.trig_poly_state_eq_constr(x)])

    # physical input constraints:
    # u ∈ [u_min, u_max]
    u_min = -10
    u_max = 10
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

    # initialize the CLF and CBF:
    (A, B) = pendulum.trig_poly_linearized_dynamics(x)
    Q = np.eye(3)
    R = np.eye(1)
    F = np.array([[0, 2, 0]])
    _, S_lqr = controllers.LinearQuadraticRegulator(A, B, Q, R, F=F)
    V_init = 0.1 * sym.Polynomial(np.dot(x, np.dot(S_lqr, x)))
    h_init = np.array([sym.Polynomial(0 * x[0] + x[1] + (1 - 0.9))])
    exclude_sets = [
        clf_cbf.ExcludeSet(
            np.array([
                sym.Polynomial(0*x[0] + x[1] + (1 - np.cos(np.pi/2)))
                ])
        ),
    ]
    within_set = clf_cbf.WithinSet(
        np.array(
            [
                -sym.Polynomial(0 * x[0] + x[1] + (1 - np.cos(np.pi / 2))),
            ]
        )
    )

    # specify the cbf relative degrees, kappa_v and kappa_h:
    # relative degrees should be a list, the size of the list is equal
    # to the number of cbfs. kappa_v is just a float for CLF constraint
    # kappa_h is a list of list of floats. The length of betas should be
    # equal to the number of cbfs. The length of each element of kappa_h
    # should be equal to the relative degree of the corresponding cbf.
    relative_degrees = [2]
    kappaV = 1
    beta = 10
    kappah = [[beta, beta]]
    barrier_eps = np.array([0.0])

    # specify clf and cbf degrees, also the state variables for the cbfs.
    # Since HOCBFs are not depend on full states, specifying the state
    # variables for the hocbfs could help to speed-up the synthesis.
    max_iter = 15
    clf_degree = 2
    cbf_degrees = [1]
    cbf_states = [
        sym.Variables(np.array([x[1]])),
    ]

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
        cbf_states=cbf_states,
        num_cbf=1,
        high_order_cbf=True,
        relative_degrees=relative_degrees,
        with_clf=True,
        use_y_squared=True,
        state_eq_constraints=state_eq_constr,
    )

    # specify compatible lagrangian degrees:
    if use_v_rep:
        compatible_lagrangian_degrees = clf_cbf.CompatibleWVrepLagrangianDegrees(
            u_vertices=[clf_cbf.XYDegree(x=2, y=0) for _ in range(u_vertices.shape[0])],
            u_extreme_rays=None,
            y=None,
            y_cross=None,
            rho_minus_V=clf_cbf.XYDegree(x=4, y=2),
            h_plus_eps=[clf_cbf.XYDegree(x=2, y=2)],
            lower_lie_derivative=[[clf_cbf.XYDegree(x=2, y=2)]],
            state_eq_constraints=[clf_cbf.XYDegree(x=2, y=2)],
        )
    else:
        compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
            lambda_y=[clf_cbf.XYDegree(x=2, y=0)],
            xi_y=clf_cbf.XYDegree(x=2, y=0),
            y=None,
            y_cross=None,
            rho_minus_V=clf_cbf.XYDegree(x=2, y=2),
            h_plus_eps=[clf_cbf.XYDegree(x=2, y=2)],
            lower_lie_derivative=[[clf_cbf.XYDegree(x=2, y=2)]],
            state_eq_constraints=[clf_cbf.XYDegree(x=2, y=2)],
        )

    # specify the safety lagrangian degrees:
    safety_lagrangian_degrees = clf_cbf.SafetySetLagrangianDegrees(
        exclude=[
            clf_cbf.ExcludeRegionLagrangianDegrees(
                cbf=[2], unsafe_region=[2], state_eq_constraints=[0]
            )
        ],
        within=[
            clf_cbf.WithinRegionLagrangianDegrees(
                cbf=[2], safe_region=2, state_eq_constraints=[0]
            )
        ],
    )

    # specifying the candidate states.
    theta = np.pi / 3
    theta_dot = 0.3
    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=np.array(
            [
                [np.sin(theta), np.cos(theta) - 1, theta_dot],
                [np.sin(theta), np.cos(theta) - 1, -theta_dot],
                [np.sin(-theta), np.cos(-theta) - 1, theta_dot],
                [np.sin(-theta), np.cos(-theta) - 1, -theta_dot],
            ]
        ),
        anchor_states=np.array([[0, 0, 0]]),
        h_anchor_bounds=[(np.array([0]), np.array([3]))],
        weight_V=1.0,
        weight_h=np.array([1.0]),
        # since this is a high relative degree system, we also need to specify
        # the followiing parameters:
        relative_degrees=relative_degrees,
        weight_lower_lie_derivatives=[
            np.array([1.0]),
        ],
        V_margin=0.8,
        h_margins=np.array([0.0, 0.0]),
    )

    # synthesis:
    V, h = compatible.bilinear_alternation(
        V_init=V_init,
        h_init=h_init,
        compatible_lagrangian_degrees=compatible_lagrangian_degrees,
        safety_sets_lagrangian_degrees=safety_lagrangian_degrees,
        kappa_V=kappaV,
        kappa_h=kappah,
        barrier_eps=barrier_eps,
        x_equilibrium=np.array([0, 0, 0]),
        clf_degree=clf_degree,
        cbf_degrees=cbf_degrees,
        max_iter=max_iter,
        record_time=True,
        compatible_states_options=compatible_states_options
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
            x_set=x,
            kappaV=kappaV,
            kappah=kappah,
        )


if __name__ == "__main__":
    main(with_u_bound=True, use_v_rep=True, save_results=False)
    # we can set use_v_rep to True or False to see the difference
    # H-rep and V-rep in synthesis time and verification time.
    # Set save_results to True to save the synthesis results, False otherwise.
