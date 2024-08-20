"""
Certify the compatible CLF/CBF without input limits.
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
    safety_sets = [
        clf_cbf.SafetySet(exclude=np.array([sym.Polynomial(x[6] + 0.5)]), within=None)
    ]
    state_eq_constraints = quadrotor.equality_constraint(x)
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        safety_sets=safety_sets,
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

    load_clf_cbf = True
    if load_clf_cbf:
        data = clf_cbf.load_clf_cbf(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../data/quadrotor_clf_cbf3.pkl",
            ),
            x_set,
        )
        V_init = data["V"]
        b_init = data["b"]
    kappa_V_sequences = [0, 1e-3, 1e-3, 1e-3, 1e-3]
    kappa_b_sequences = [np.array([kappa_V]) for kappa_V in kappa_V_sequences]

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
    safety_sets_lagrangian_degrees = [
        clf_cbf.SafetySetLagrangianDegrees(
            exclude=clf_cbf.ExcludeRegionLagrangianDegrees(
                cbf=0, unsafe_region=[0], state_eq_constraints=[0]
            ),
            within=None,
        )
    ]
    barrier_eps = np.array([0.000])
    x_equilibrium = np.zeros((13,))
    candidate_compatible_states_sequences = []
    candidate_compatible_states_sequences.append(np.zeros((6, 13)))
    candidate_compatible_states_sequences[0][1, 6] = -0.3
    candidate_compatible_states_sequences[0][2, 5] = -0.3
    candidate_compatible_states_sequences[0][2, 6] = -0.3
    candidate_compatible_states_sequences[0][3, 5] = 0.3
    candidate_compatible_states_sequences[0][3, 6] = -0.3
    candidate_compatible_states_sequences[0][4, 4] = 0.3
    candidate_compatible_states_sequences[0][4, 6] = -0.3
    candidate_compatible_states_sequences[0][5, 4] = -0.3
    candidate_compatible_states_sequences[0][5, 6] = -0.3

    candidate_compatible_states_sequences.append(np.zeros((6, 13)))
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = -0.4
    candidate_compatible_states_sequences[-1][2, 6] = -0.3
    candidate_compatible_states_sequences[-1][3, 5] = 0.4
    candidate_compatible_states_sequences[-1][3, 6] = -0.3
    candidate_compatible_states_sequences[-1][4, 4] = 0.3
    candidate_compatible_states_sequences[-1][4, 6] = -0.4
    candidate_compatible_states_sequences[-1][5, 4] = -0.3
    candidate_compatible_states_sequences[-1][5, 6] = -0.4

    candidate_compatible_states_sequences.append(np.zeros((6, 13)))
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = -0.5
    candidate_compatible_states_sequences[-1][2, 6] = -0.35
    candidate_compatible_states_sequences[-1][3, 5] = 0.5
    candidate_compatible_states_sequences[-1][3, 6] = -0.35
    candidate_compatible_states_sequences[-1][4, 4] = 0.5
    candidate_compatible_states_sequences[-1][4, 6] = -0.35
    candidate_compatible_states_sequences[-1][5, 4] = -0.5
    candidate_compatible_states_sequences[-1][5, 6] = -0.35

    candidate_compatible_states_sequences.append(np.zeros((6, 13)))
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = -0.6
    candidate_compatible_states_sequences[-1][2, 6] = -0.35
    candidate_compatible_states_sequences[-1][3, 5] = 0.6
    candidate_compatible_states_sequences[-1][3, 6] = -0.35
    candidate_compatible_states_sequences[-1][4, 4] = 0.6
    candidate_compatible_states_sequences[-1][4, 6] = -0.35
    candidate_compatible_states_sequences[-1][5, 4] = -0.6
    candidate_compatible_states_sequences[-1][5, 6] = -0.35

    candidate_compatible_states_sequences.append(np.zeros((5, 13)))
    candidate_compatible_states_sequences[-1][1, 5] = -0.7
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = 0.7
    candidate_compatible_states_sequences[-1][2, 6] = -0.35
    candidate_compatible_states_sequences[-1][3, 4] = 0.7
    candidate_compatible_states_sequences[-1][3, 6] = -0.35
    candidate_compatible_states_sequences[-1][4, 4] = -0.7
    candidate_compatible_states_sequences[-1][4, 6] = -0.35

    lagrangian_sos_types = [
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ]
    b_margins_sequence = [
        None,
        None,
        np.array([0.05]),
        np.array([0.05]),
        np.array([0.02]),
    ]
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)
    V = V_init
    b = b_init
    max_iter_sequence = [1, 2, 1, 2, 1]
    backoff_scale_sequence = [
        BackoffScale(rel=None, abs=0.2),
        BackoffScale(rel=None, abs=0.2),
        BackoffScale(rel=None, abs=0.2),
        BackoffScale(rel=None, abs=0.1),
        BackoffScale(rel=None, abs=0.2),
    ]
    for i in range(len(candidate_compatible_states_sequences)):
        compatible_states_options = clf_cbf.CompatibleStatesOptions(
            candidate_compatible_states=candidate_compatible_states_sequences[i],
            anchor_states=np.zeros((1, 13)),
            b_anchor_bounds=[(np.array([0.6]), np.array([1.0]))],
            weight_V=1,
            weight_b=np.array([1]),
            b_margins=b_margins_sequence[i],
        )

        kappa_V = kappa_V_sequences[i]
        kappa_b = kappa_b_sequences[i]
        V, b = compatible.bilinear_alternation(
            V,
            b,
            compatible_lagrangian_degrees,
            safety_sets_lagrangian_degrees,
            kappa_V,
            kappa_b,
            barrier_eps,
            x_equilibrium,
            V_degree,
            b_degrees,
            max_iter=max_iter_sequence[i],
            solver_options=solver_options,
            lagrangian_coefficient_tol=None,
            compatible_states_options=compatible_states_options,
            backoff_scale=backoff_scale_sequence[i],
            lagrangian_sos_type=lagrangian_sos_types[i],
        )
        pickle_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../../data/quadrotor_clf_cbf{i}.pkl",
        )
        clf_cbf.save_clf_cbf(V, b, x_set, kappa_V, kappa_b, pickle_path)


if __name__ == "__main__":
    main(use_y_squared=True, with_u_bound=False)