"""
Synthesize compatible CLF/CBF for power converter.
"""

import os

import numpy as np

import pydrake.symbolic as sym
import pydrake.systems.controllers
import pydrake.solvers as solvers

import compatible_clf_cbf.clf as clf
import compatible_clf_cbf.clf_cbf as clf_cbf
import examples.power_converter.plant as plant


def lqr():
    A, B = plant.linearize_dynamics(np.zeros((3,)), np.zeros((2,)))
    K, S = pydrake.systems.controllers.LinearQuadraticRegulator(
        A, B, Q=np.eye(3, 3), R=np.eye(2, 2)
    )
    return K, S


def find_regional_clf(
    V_degree: int, x: np.ndarray, save_pickle_filename="power_converter_V_init.pkl"
) -> sym.Polynomial:
    K_lqr, S_lqr = lqr()
    u_lqr = -K_lqr @ x
    dynamics_expr = plant.dynamics(x, u_lqr)
    dynamics = np.array([sym.Polynomial(dynamics_expr[i]) for i in range(3)])
    positivity_eps = 0.01
    d = V_degree / 2
    kappa = 0.1
    state_eq_constraints = np.empty((0,))
    positivity_ceq_lagrangian_degrees = []
    derivative_ceq_lagrangian_degrees = []
    state_ineq_constraints = np.array([sym.Polynomial(x.dot(x) - 1e-3)])
    positivity_cin_lagrangian_degrees = [V_degree - 2]
    derivative_cin_lagrangian_degrees = [int(np.ceil((V_degree + 1) / 2) * 2 - 2)]

    prog, V = clf.find_candidate_regional_lyapunov(
        x,
        dynamics,
        V_degree,
        positivity_eps,
        d,
        kappa,
        state_eq_constraints,
        positivity_ceq_lagrangian_degrees,
        derivative_ceq_lagrangian_degrees,
        state_ineq_constraints,
        positivity_cin_lagrangian_degrees,
        derivative_cin_lagrangian_degrees,
    )
    result = solvers.Solve(prog)
    assert result.is_success()
    V_sol = result.GetSolution(V)
    x_set = sym.Variables(x)
    pickle_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", save_pickle_filename
    )
    clf.save_clf(V_sol, x_set, kappa, pickle_path)
    return V_sol


def search(use_y_squared: bool):
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = plant.affine_dynamics(x)

    safety_sets = [
        clf_cbf.SafetySet(
            exclude=None,
            within=np.array(
                [
                    sym.Polynomial(x[0] - 0.2),
                    sym.Polynomial(-x[0] - 0.8),
                    sym.Polynomial(((x[1] - 0.001) ** 2) + x[2] ** 2 - 1.2**2),
                ]
            ),
        )
    ]

    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        safety_sets=safety_sets,
        Au=None,
        bu=None,
        with_clf=True,
        use_y_squared=use_y_squared,
    )
    V_degree = 2
    V_init = 50 * find_regional_clf(V_degree, x)
    b_init = np.array([1 - V_init])

    kappa_V = 1e-3
    kappa_b = np.array([kappa_V])

    compatible_lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[
            clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0 if use_y_squared else 1)
            for _ in range(2)
        ],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(
            x=2, y=0 if use_y_squared else 1
        ),
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

    safety_sets_lagrangian_degrees = [
        clf_cbf.SafetySetLagrangianDegrees(
            exclude=None,
            within=[
                clf_cbf.WithinRegionLagrangianDegrees(
                    cbf=0, safe_region=0, state_eq_constraints=None
                )
                for _ in range(3)
            ],
        )
    ]
    barrier_eps = np.array([0.0])
    x_equilibrium = np.zeros((3,))

    candidate_compatible_states = np.array(
        [[0, 1, 0.5], [0, 1, -0.5], [-0.3, 0.5, 0.5], [-0.3, -0.5, -0.5]]
    )
    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=candidate_compatible_states,
        anchor_states=np.zeros((1, 3)),
        b_anchor_bounds=[[np.array([0.1]), np.array([1])]],
        weight_V=1,
        weight_b=np.array([1]),
        V_margin=0,
        b_margins=np.array([0.01]),
    )

    kappa_V = 1e-3
    kappa_b = np.array([kappa_V])

    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 1)

    compatible.bilinear_alternation(
        V_init=V_init,
        b_init=b_init,
        compatible_lagrangian_degrees=compatible_lagrangian_degrees,
        safety_sets_lagrangian_degrees=safety_sets_lagrangian_degrees,
        kappa_V=kappa_V,
        kappa_b=kappa_b,
        barrier_eps=barrier_eps,
        x_equilibrium=x_equilibrium,
        clf_degree=2,
        cbf_degrees=[2],
        max_iter=5,
        solver_options=solver_options,
        compatible_states_options=compatible_states_options,
        # backoff_scale=BackoffScale(abs=10, rel=None),
        # lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSdsos
    )


def main():
    search(use_y_squared=True)


if __name__ == "__main__":
    main()
