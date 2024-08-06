"""
Certify the compatible CLF/CBF with input limits.
This uses the taylor expansion of the 2D quadrotor dynamics.
"""

from enum import Enum
from typing import Tuple

import numpy as np

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.controllers

from compatible_clf_cbf import clf_cbf
from compatible_clf_cbf import clf
import compatible_clf_cbf.utils
from examples.quadrotor2d.plant import Quadrotor2dPlant


class GrowHeuristics(Enum):
    kInnerEllipsoid = 1
    kCompatibleStates = 2


def lqr(quadrotor: Quadrotor2dPlant) -> Tuple[np.ndarray, np.ndarray]:
    x_des = np.zeros((6,))
    u_des = np.full((2,), quadrotor.m * quadrotor.g / 2)
    xdot_des = quadrotor.dynamics(x_des, u_des)
    np.testing.assert_allclose(xdot_des, np.zeros((6,)), atol=1e-10)
    A, B = quadrotor.linearize_dynamics(x_des, u_des)
    Q = np.diag([1, 1, 1, 10, 10, 10])
    R = np.diag([10.0, 10])
    K, S = pydrake.systems.controllers.LinearQuadraticRegulator(A, B, Q, R)
    return K, S


def search_clf_cbf(
    use_y_squared: bool, with_u_bound: bool, grow_heuristics: GrowHeuristics
):
    x = sym.MakeVectorContinuousVariable(6, "x")
    quadrotor = Quadrotor2dPlant()
    f, g = quadrotor.taylor_affine_dynamics(x)

    if with_u_bound:
        Au = np.concatenate((np.eye(2), -np.eye(2)), axis=0)
        u_bound = quadrotor.m * quadrotor.g * 1.5
        bu = np.concatenate((np.full((2,), u_bound), np.zeros((2,))))
    else:
        Au, bu = None, None

    K_lqr, S_lqr = lqr(quadrotor)
    V_init = sym.Polynomial(x.dot(S_lqr @ x) / 0.01)
    V_init = V_init.RemoveTermsWithSmallCoefficients(1e-10)
    b_init = np.array([1 - V_init])
    kappa_V = 1e-3
    kappa_b = np.array([kappa_V])

    search_clf = False
    if search_clf:
        clf_search = clf.ControlLyapunov(
            f=f,
            g=g,
            x=x,
            x_equilibrium=np.zeros((6,)),
            u_vertices=None,
            state_eq_constraints=None,
        )
        clf_lagrangian_degrees = clf.ClfWoInputLimitLagrangianDegrees(
            dVdx_times_f=4,
            dVdx_times_g=[2, 2],
            rho_minus_V=4,
            state_eq_constraints=None,
        )
        solver_options = solvers.SolverOptions()
        solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)
        clf_search.search_lagrangian_given_clf(
            V_init,
            rho=1,
            kappa=kappa_V,
            lagrangian_degrees=clf_lagrangian_degrees,
            solver_options=solver_options,
        )
        return

    # Ground as the unsafe region.
    unsafe_regions = [np.array([sym.Polynomial(x[1] + 0.5)])]
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        unsafe_regions=unsafe_regions,
        Au=Au,
        bu=bu,
        with_clf=True,
        use_y_squared=use_y_squared,
        state_eq_constraints=None,
    )
    lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[
            clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0) for _ in range(2)
        ],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(x=4, y=0),
        y=(
            None
            if use_y_squared
            else [
                clf_cbf.CompatibleLagrangianDegrees.Degree(x=4, y=0)
                for _ in range(compatible.y.size)
            ]
        ),
        rho_minus_V=clf_cbf.CompatibleLagrangianDegrees.Degree(x=4, y=2),
        b_plus_eps=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=4, y=2)],
        state_eq_constraints=None,
    )
    barrier_eps = np.array([0.0])

    unsafe_regions_lagrangian_degrees = [
        clf_cbf.UnsafeRegionLagrangianDegrees(
            cbf=0, unsafe_region=[2], state_eq_constraints=None
        )
    ]
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)
    if grow_heuristics == GrowHeuristics.kInnerEllipsoid:
        inner_ellipsoid_options = clf_cbf.InnerEllipsoidOptions(
            x_inner=np.zeros((6,)),
            ellipsoid_trust_region=200,
            find_inner_ellipsoid_max_iter=3,
        )
        binary_search_scale_options = compatible_clf_cbf.utils.BinarySearchOptions(
            min=0.95, max=3, tol=1e-1
        )
        compatible_states_options = None
        max_iter = 3
    elif grow_heuristics == GrowHeuristics.kCompatibleStates:
        inner_ellipsoid_options = None
        binary_search_scale_options = None
        compatible_states_options = clf_cbf.CompatibleStatesOptions(
            candidate_compatible_states=np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, -0.25, 0, 0, 0, 0],
                    [0.2, -0.25, 0, 0, 0, 0],
                    [-0.2, -0.25, 0, 0, 0, 0],
                ]
            ),
            anchor_states=np.zeros((1, 6)),
            b_anchor_bounds=[(np.array([0.0]), np.array([1]))],
            weight_V=1,
            weight_b=np.array([1]),
        )
        max_iter = 5
    else:
        raise Exception("unsupported grow heuristics.")

    V, b = compatible.bilinear_alternation(
        V_init,
        b_init,
        lagrangian_degrees,
        unsafe_regions_lagrangian_degrees,
        kappa_V,
        kappa_b,
        barrier_eps,
        x_equilibrium=np.zeros((6,)),
        clf_degree=2,
        cbf_degrees=[2],
        max_iter=max_iter,
        solver_options=solver_options,
        inner_ellipsoid_options=inner_ellipsoid_options,
        binary_search_scale_options=binary_search_scale_options,
        compatible_states_options=compatible_states_options,
        backoff_scale=0.02,
    )
    return


def main():
    search_clf_cbf(
        use_y_squared=True,
        with_u_bound=False,
        grow_heuristics=GrowHeuristics.kCompatibleStates,
    )
    # SDP with u_bound is a lot slower than without u_bound.
    # search_clf_cbf(
    #    use_y_squared=True,
    #    with_u_bound=True,
    #    grow_heuristics=GrowHeuristics.kInnerEllipsoid,
    # )


if __name__ == "__main__":
    main()
