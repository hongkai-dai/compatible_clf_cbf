"""
Search for a CLF.
This uses the polynomial dynamics of the 2D quadrotor (with sin/cos as part of
the state), and equality constraint to enforce the unit-length constraint on the
cos/sin.
"""

from typing import Tuple

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.controllers

from compatible_clf_cbf import clf
from examples.quadrotor2d.plant import Quadrotor2dTrigPlant


def lqr(quadrotor: Quadrotor2dTrigPlant) -> Tuple[np.ndarray, np.ndarray]:
    x_des = np.zeros((7,))
    u_des = np.full((2,), quadrotor.m * quadrotor.g / 2)
    xdot_des = quadrotor.dynamics(x_des, u_des)
    np.testing.assert_allclose(xdot_des, np.zeros((7,)), atol=1e-10)

    A, B = quadrotor.linearize_dynamics(x_des, u_des)
    Q = np.diag([1, 1, 10, 10, 10, 10, 10.0])
    R = np.diag([10.0, 10])
    # Gradient of the constraint sin^2 + cos^2 = 1
    F = np.array([[0, 0, 0, 2, 0, 0, 0]])
    K, S = pydrake.systems.controllers.LinearQuadraticRegulator(
        A, B, Q, R, N=np.empty((0, 2)), F=F
    )
    return K, S


def find_trig_regional_clf(V_degree: int, x: np.ndarray) -> sym.Polynomial:
    quadrotor = Quadrotor2dTrigPlant()
    K_lqr, _ = lqr(quadrotor)
    u_lqr = -K_lqr @ x + np.full((2,), quadrotor.m * quadrotor.g / 2)
    dynamics_expr = quadrotor.dynamics(x, u_lqr)
    dynamics = np.array([sym.Polynomial(dynamics_expr[i]) for i in range(7)])
    positivity_eps = 0.01
    d = int(V_degree / 2)
    kappa = 1e-4
    state_eq_constraints = quadrotor.equality_constraint(x)
    positivity_ceq_lagrangian_degrees = [V_degree - 2]
    derivative_ceq_lagrangian_degrees = [int(np.ceil((V_degree + 1) / 2) * 2 - 2)]
    state_ineq_constraints = np.array([sym.Polynomial(x.dot(x) - 1e-4)])
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
    return V_sol


def main(with_u_bound: bool):
    x = sym.MakeVectorContinuousVariable(7, "x")
    quadrotor = Quadrotor2dTrigPlant()
    f, g = quadrotor.affine_dynamics(x)

    if with_u_bound:
        thrust_max = 1.5 * quadrotor.m * quadrotor.g
        u_vertices = np.array(
            [[0, 0], [0, thrust_max], [thrust_max, 0], [thrust_max, thrust_max]]
        )
    else:
        u_vertices = None

    state_eq_constraints = quadrotor.equality_constraint(x)

    V_degree = 2
    V_init = find_trig_regional_clf(V_degree, x)
    kappa_V = 1e-4
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)

    clf_search = clf.ControlLyapunov(
        f=f,
        g=g,
        x=x,
        x_equilibrium=np.zeros((7,)),
        u_vertices=u_vertices,
        state_eq_constraints=state_eq_constraints,
    )
    if with_u_bound:
        clf_lagrangian_degrees = clf.ClfWInputLimitLagrangianDegrees(
            V_minus_rho=0, Vdot=[2, 2, 2, 2], state_eq_constraints=[4]
        )
    else:
        clf_lagrangian_degrees = clf.ClfWoInputLimitLagrangianDegrees(
            dVdx_times_f=4, dVdx_times_g=[3, 3], rho_minus_V=4, state_eq_constraints=[4]
        )
    clf_search.search_lagrangian_given_clf(
        V_init,
        rho=1,
        kappa=kappa_V,
        lagrangian_degrees=clf_lagrangian_degrees,
        solver_options=solver_options,
    )
    return


if __name__ == "__main__":
    main(with_u_bound=True)
