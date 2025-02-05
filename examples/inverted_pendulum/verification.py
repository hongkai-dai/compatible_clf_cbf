import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

import itertools
import numpy as np
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers

from compatible_clf_cbf import clf_hocbf
from compatible_clf_cbf.utils import system_linearization
from examples.inverted_pendulum import pendulum

def main(with_u_bound:bool, use_v_rep: bool):
    # system model:
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = pendulum.affine_trig_poly_dynamics(x)
    state_eq_constr = np.array([pendulum.affine_trig_poly_state_constraints(x)])

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
    

    # compute the V(x) using the LQR cost to go function. This could avoid the issue of 
    # non-uniform relative degree. In this example, since we are using the non-linear 
    # system, we need to linearize it near the equilibrium point. 
    # accroding to the definition of the extended dynamics:
    # the equilibrium point should be:
    # x0 = 0, x1 = 0, x2 = 0.
    x_eq = np.array([0.0, 0.0, 0.0])
    u_eq = np.array([0.0])
    eq_point = (x_eq, u_eq)
    (A, B) = system_linearization(f=f, g=g, states=x, eq_point=eq_point)
    Q = np.eye(3)
    R = np.eye(1)
    F = np.array([[0,2,0]])
    _, S_lqr = controllers.LinearQuadraticRegulator(A, B, Q, R, F=F)


    # specify the V(x), b(x) and unsafe regions:
    # V(x) should be Polynomial
    # h(x) should be an array of polynomials specifying all the cbfs.
    # the real unsafe region is \theta \in [-pi/2, pi/2].
    V = 0.1*sym.Polynomial(np.dot(x, np.dot(S_lqr, x)))
    h = np.array([
        sym.Polynomial(0*x[0] + x[1] + (1 - 0.9))
        ])
    exclude_sets = [
        # clf_cbf.ExcludeSet(
        #     np.array([
        #         sym.Polynomial(0*x[0] + x[1] + (1 - np.cos(np.pi/2)))
        #         ])
        # ),
    ]
    within_set = clf_hocbf.WithinSet(
        np.array([
            -sym.Polynomial(0*x[0] + x[1] + (1 - np.cos(np.pi/2))),
        ])
    )

    # specify the cbf relative degrees, alpha and betas:
    # relative degrees are put in a list, the size of the list should be equal to the number of cbfs
    # alpha is just a float for CLF constraint
    # betas are put in list of list of floats, the outter number of elements should be the same as 
    # the number of cbf, the inner number of elements should be the same as relative degrees. 
    relative_degrees = [2]
    kappaV = 1
    beta = 10
    kappah = [
        [beta, beta]
        ]
        
    # create comaptibleCLFCBF object:
    compatible = clf_hocbf.CompatibleClfCbf(
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
        with_clf=True,
        use_y_squared=True,
        state_eq_constraints=state_eq_constr
    )

    # specify compatible lagrangian degrees:
    if use_v_rep:
        compatible_lagrangian_degrees = clf_hocbf.CompatibleWVrepLagrangianDegrees(
            u_vertices=[clf_hocbf.XYDegree(x=2, y=0) for _ in range(u_vertices.shape[0])],
            u_extreme_rays=None,
            xi_y=None,
            y=None,
            rho_minus_V=clf_hocbf.XYDegree(x=4, y=2),
            h_plus_eps=[
                clf_hocbf.XYDegree(x=2, y=2)
                ],
            lower_lie_derivatives=[
                [clf_hocbf.XYDegree(x=2, y=2)]
                ],
            state_eq_constraints=[
                clf_hocbf.XYDegree(x=2, y=2)
            ],
        )
    else:
        compatible_lagrangian_degrees = clf_hocbf.CompatibleLagrangianDegrees(
            lambda_y=[
                clf_hocbf.XYDegree(x=2, y=0)
                ],
            xi_y=clf_hocbf.XYDegree(x=2, y=0),
            y=None,
            rho_minus_V=clf_hocbf.XYDegree(x=2, y=2),
            h_plus_eps = [clf_hocbf.XYDegree(x=2, y=2)],
            lower_lie_derivatives=[
                [clf_hocbf.XYDegree(x=2, y=2)]
                ],
            state_eq_constraints=[
                clf_hocbf.XYDegree(x=2, y=2)
            ]
        )
    
    # specify the safety lagrangian degrees:
    safety_lagrangian_degrees = clf_hocbf.SafetySetLagrangianDegrees(
        exclude=[
            # clf_cbf.ExcludeRegionLagrangianDegrees(
            #     cbf=[2], unsafe_region=[2], state_eq_constraints=[0]
            # )
        ],
        within=[
            clf_hocbf.WithinRegionLagrangianDegrees(
                cbf=[2], safe_region=2, state_eq_constraints=[0]
            )
        ]
    )

    barrier_eps = np.array([0.0])

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
        relative_degrees=relative_degrees,
        compatible_lagrangian_degrees=compatible_lagrangian_degrees,
        safety_set_lagrangian_degrees=safety_lagrangian_degrees,
        record_time=True,
    )

    assert compatible_lagrangians is not None
    assert safety_sets_lagrangians is not None
    

if __name__ == "__main__":
    main(with_u_bound=True, use_v_rep=False)
    print("verification passed!")