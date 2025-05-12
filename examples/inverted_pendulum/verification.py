import time
import itertools
import numpy as np
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers

from compatible_clf_cbf import clf_cbf
from examples.inverted_pendulum import plant


def main(with_u_bound: bool, use_v_rep: bool):
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

    # specify the V(x), b(x) and unsafe regions:
    # V(x) should be symbolic Polynomial
    # h(x) should be an array of symbolic polynomials.
    # the real unsafe region is \theta \in [-pi/2, pi/2].
    V = 0.1 * sym.Polynomial(np.dot(x, np.dot(S_lqr, x)))
    h = np.array([sym.Polynomial(0 * x[0] + x[1] + (1 - 0.9))])
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
    # note that here the exclude_set and within_set actually identify the
    # same safe region. We define them repeatedly to ensure
    # all the functionalities are working.

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
        num_cbf=1,
        high_order_cbf=True,
        relative_degrees=relative_degrees,
        cbf_states=[sym.Variables(x[0:2])],
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

    barrier_eps = np.array([0.0])

    start_time = time.time()
    # verification:
    (
        compatible_lagrangians,
        safety_sets_lagrangians,
    ) = compatible.search_lagrangians_given_clf_cbf(
        V=V,
        h=h,
        kappa_V=kappaV,
        kappa_h=kappah,
        barrier_eps=barrier_eps,
        compatible_lagrangian_degrees=compatible_lagrangian_degrees,
        safety_set_lagrangian_degrees=safety_lagrangian_degrees,
    )

    end_time = time.time()
    assert compatible_lagrangians is not None
    assert safety_sets_lagrangians is not None
    print("compatible verification time: ", end_time - start_time)


if __name__ == "__main__":
    main(with_u_bound=True, use_v_rep=False)
    # we can set the use_v_rep to True or False to see the difference
    # of verification time of H-rep and V-rep in this example.
