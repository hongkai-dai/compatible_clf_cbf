import time
import itertools
import numpy as np
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers

from compatible_clf_cbf import clf_cbf
from examples.adaptive_cruise_control import plant


def main(with_u_bound: bool, use_v_rep: bool):
    # system model:
    x = sym.MakeVectorContinuousVariable(2, "x")
    cruise_control = plant.AdaptiveCruiseControlPlant()
    (f, g) = cruise_control.affine_dynamics(x)

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
    A, B = cruise_control.linear_dynamics(x)
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
    V = sym.Polynomial(np.dot(x, np.dot(S_lqr, x)))
    h = np.array([sym.Polynomial(-x[1] + 0.5), sym.Polynomial(x[1] - (-0.5))])
    exclude_sets = [
        clf_cbf.ExcludeSet(
            np.array(
                [sym.Polynomial(-x[1] + (cruise_control.d_max - cruise_control.d_ref))]
            )
        ),
        clf_cbf.ExcludeSet(
            np.array(
                [sym.Polynomial(x[1] - (cruise_control.d_min - cruise_control.d_ref))]
            )
        ),
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
    # the same as the number of cbf, the inner number of elements should be the same
    # as relative degrees.
    relative_degrees = [2, 2]
    kappaV = 1
    kappah = [[1, 1], [1, 1]]

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
        cbf_states=[
            sym.Variables(np.array([x[1]])),
            sym.Variables(np.array([x[1]]))
            ],
        with_clf=True,
        use_y_squared=True,
        state_eq_constraints=None,
    )

    # specify lagrangian degrees:
    if use_v_rep:
        compatible_lagrangian_degrees = clf_cbf.CompatibleWVrepLagrangianDegrees(
            u_vertices=[clf_cbf.XYDegree(x=2, y=0) for _ in range(u_vertices.shape[0])],
            u_extreme_rays=None,
            y=None,
            y_cross=None,
            rho_minus_V=clf_cbf.XYDegree(x=4, y=2),
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
            clf_cbf.ExcludeRegionLagrangianDegrees(
                cbf=[0, 0], unsafe_region=[0], state_eq_constraints=None
            ),
            clf_cbf.ExcludeRegionLagrangianDegrees(
                cbf=[0, 0], unsafe_region=[0], state_eq_constraints=None
            ),
        ],
        within=[
            clf_cbf.WithinRegionLagrangianDegrees(
                cbf=[0, 0], safe_region=0, state_eq_constraints=None
            ),
            clf_cbf.WithinRegionLagrangianDegrees(
                cbf=[0, 0], safe_region=0, state_eq_constraints=None
            ),
        ],
    )

    # Solve the SOS programe:
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
        safety_set_lagrangian_degrees=safety_lagrangian_degrees
    )

    end_time = time.time()
    assert compatible_lagrangians is not None
    assert safety_sets_lagrangians is not None
    print("Compatibility verification time: ", end_time - start_time)


if __name__ == "__main__":
    main(with_u_bound=True, use_v_rep=False)
    # we can set the use_v_rep to True or False to see the difference
    # of verification time of H-rep and V-rep in this example.
