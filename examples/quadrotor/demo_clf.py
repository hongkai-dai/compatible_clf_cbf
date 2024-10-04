"""
Search for a CLF.
This uses the polynomial dynamics of the 3D quadrotor (with quaternion as part
of the state), and the equality constraint to enforce the unit length constraint
on the quaternion.
"""

import itertools
import os
from typing import Tuple

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.controllers

from compatible_clf_cbf import clf
from examples.quadrotor.plant import QuadrotorPolyPlant


def lqr(quadrotor: QuadrotorPolyPlant) -> Tuple[np.ndarray, np.ndarray]:
    x_des = np.zeros((13,))
    u_des = np.full((4,), quadrotor.m * quadrotor.g / 4)
    xdot_des = quadrotor.dynamics(x_des, u_des)
    np.testing.assert_allclose(xdot_des, np.zeros((13,)), atol=1e-8)

    A, B = quadrotor.linearize_dynamics(x_des, u_des)
    Q = np.diag([1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10])
    R = np.diag([10, 10, 10, 10])
    # Gradient of the constraint quaternion.norm() = 1
    F = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    K, S = pydrake.systems.controllers.LinearQuadraticRegulator(
        A, B, Q, R, N=np.empty((0, 4)), F=F
    )
    return K, S


def find_trig_regional_clf(
    V_degree: int, x: np.ndarray, save_pickle_filename="quadrotor_V_init.pkl"
) -> sym.Polynomial:
    quadrotor = QuadrotorPolyPlant()
    K_lqr, _ = lqr(quadrotor)
    u_lqr = -K_lqr @ x + np.full((4,), quadrotor.m * quadrotor.g / 4)
    dynamics_expr = quadrotor.dynamics(x, u_lqr)
    dynamics = np.array([sym.Polynomial(dynamics_expr[i]) for i in range(13)])
    positivity_eps = 0.01
    d = int(V_degree / 2)
    kappa = 0.1
    state_eq_constraints = quadrotor.equality_constraint(x)
    positivity_ceq_lagrangian_degrees = [V_degree - 2]
    derivative_ceq_lagrangian_degrees = [int(np.ceil((V_degree + 3) / 2) * 2 - 2)]
    state_ineq_constraints = np.array([sym.Polynomial(x.dot(x) - 1e-3)])
    positivity_cin_lagrangian_degrees = [V_degree - 2]
    derivative_cin_lagrangian_degrees = derivative_ceq_lagrangian_degrees

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


def main(with_u_bound: bool):
    x = sym.MakeVectorContinuousVariable(13, "x")
    quadrotor = QuadrotorPolyPlant()
    f, g = quadrotor.affine_dynamics(x)

    if with_u_bound:
        thrust_max = quadrotor.m * quadrotor.g
        u_vertices = np.array(list(itertools.product([0, thrust_max], repeat=4)))
    else:
        u_vertices = None

    state_eq_constraints = quadrotor.equality_constraint(x)

    V_degree = 2
    print("Find regional CLF.")
    V_init = find_trig_regional_clf(V_degree, x)
    kappa_V = 0.1
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)

    clf_search = clf.ControlLyapunov(
        f=f,
        g=g,
        x=x,
        x_equilibrium=np.zeros((13,)),
        u_vertices=u_vertices,
        state_eq_constraints=state_eq_constraints,
    )
    if with_u_bound:
        clf_lagrangian_degrees = clf.ClfWInputLimitLagrangianDegrees(
            V_minus_rho=2, Vdot=[2] * 16, state_eq_constraints=[4]
        )
    else:
        clf_lagrangian_degrees = clf.ClfWoInputLimitLagrangianDegrees(
            dVdx_times_f=4,
            dVdx_times_g=[3, 3, 3, 3],
            rho_minus_V=4,
            state_eq_constraints=[4],
        )
    candidate_stable_states = np.zeros((4, 13))
    candidate_stable_states[0, 4:7] = np.array([1, 0, 0])
    candidate_stable_states[1, 4:7] = np.array([-1, 0, 0])
    candidate_stable_states[2, 4:7] = np.array([0, 1, 0])
    candidate_stable_states[3, 4:7] = np.array([0, -1, 0])
    stable_states_options = clf.StableStatesOptions(
        candidate_stable_states=candidate_stable_states, V_margin=0.01
    )
    print("Bilinear alternation.")
    V = clf_search.bilinear_alternation(
        V_init,
        clf_lagrangian_degrees,
        kappa=kappa_V,
        clf_degree=2,
        x_equilibrium=np.zeros((13,)),
        max_iter=5,
        stable_states_options=stable_states_options,
        solver_options=solver_options,
    )
    clf.save_clf(
        V,
        clf_search.x_set,
        kappa=kappa_V,
        pickle_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/quadrotor_clf.pkl"
        ),
    )
    return


if __name__ == "__main__":
    main(with_u_bound=True)
