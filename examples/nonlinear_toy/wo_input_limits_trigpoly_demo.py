"""
Find the CLF/CBF without the input limits.
We use the trigonometric state with polynomial dynamics.
"""

from enum import Enum

import numpy as np

import pydrake.symbolic as sym
import pydrake.solvers as solvers

import compatible_clf_cbf.clf_cbf as clf_cbf
import compatible_clf_cbf.utils as utils
import examples.nonlinear_toy.toy_system as toy_system


class GrowHeuristics(Enum):
    kInnerEllipsoid = 1
    kCompatibleStates = 2


def main(grow_heuristics: GrowHeuristics):
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = toy_system.affine_trig_poly_dynamics(x)
    state_eq_constraints = np.array([toy_system.affine_trig_poly_state_constraints(x)])
    use_y_squared = True
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        unsafe_regions=[np.array([sym.Polynomial(x[0] + x[1] + x[2] + 3)])],
        Au=None,
        bu=None,
        with_clf=True,
        use_y_squared=use_y_squared,
        state_eq_constraints=state_eq_constraints,
    )
    V_init = sym.Polynomial(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) / 0.1
    b_init = np.array([sym.Polynomial(0.1 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2)])

    compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0)],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0),
        y=None,
        rho_minus_V=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2),
        b_plus_eps=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2)],
        state_eq_constraints=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2)],
    )
    unsafe_region_lagrangian_degrees = [
        clf_cbf.UnsafeRegionLagrangianDegrees(
            cbf=0, unsafe_region=[0], state_eq_constraints=[0]
        )
    ]
    kappa_V = 0.01
    kappa_b = np.array([0.01])
    barrier_eps = np.array([0.001])

    x_equilibrium = np.array([0, 0.0, 0.0])

    clf_degree = 2
    cbf_degrees = [2]
    max_iter = 5

    if grow_heuristics == GrowHeuristics.kInnerEllipsoid:
        binary_search_scale_options = utils.BinarySearchOptions(min=1, max=2, tol=0.1)
        find_inner_ellipsoid_max_iter = 1
        compatible_states_options = None
        solver_options = None
    elif grow_heuristics == GrowHeuristics.kCompatibleStates:
        binary_search_scale_options = None
        find_inner_ellipsoid_max_iter = 0
        compatible_states_options = clf_cbf.CompatibleStatesOptions(
            candidate_compatible_states=np.array(
                [
                    [np.sin(np.pi / 2), np.cos(np.pi / 2) - 1, 0.5],
                    [np.sin(np.pi / 2), np.cos(np.pi / 2) - 1, -0.5],
                ]
            ),
            anchor_states=np.array([[0.0, 0, 0]]),
            b_anchor_bounds=[(np.array([0]), np.array([0.1]))],
            weight_V=1,
            weight_b=np.array([1.0]),
        )
        solver_options = solvers.SolverOptions()
        solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 0)

    V, b = compatible.bilinear_alternation(
        V_init,
        b_init,
        compatible_lagrangian_degrees,
        unsafe_region_lagrangian_degrees,
        kappa_V,
        kappa_b,
        barrier_eps,
        x_equilibrium,
        clf_degree,
        cbf_degrees,
        max_iter,
        x_inner=x_equilibrium,
        binary_search_scale_options=binary_search_scale_options,
        find_inner_ellipsoid_max_iter=find_inner_ellipsoid_max_iter,
        compatible_states_options=compatible_states_options,
        solver_options=solver_options,
    )
    print(f"V={V}")
    print(f"b={b}")
    if grow_heuristics == GrowHeuristics.kCompatibleStates:
        assert V is not None
        assert compatible_states_options is not None
        assert compatible_states_options.candidate_compatible_states is not None
        V_candidates = V.EvaluateIndeterminates(
            x, compatible_states_options.candidate_compatible_states.T
        )
        print(f"V(candidate_compatible_states)={V_candidates}")
        b_candidates = b[0].EvaluateIndeterminates(
            x, compatible_states_options.candidate_compatible_states.T
        )
        print(f"b(candidate_compatible_states)={b_candidates}")


if __name__ == "__main__":
    main(GrowHeuristics.kCompatibleStates)
