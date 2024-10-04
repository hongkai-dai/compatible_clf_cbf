import itertools
import os
from typing import Tuple

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym
from pydrake.autodiffutils import (
    AutoDiffXd,
    InitializeAutoDiff,
    ExtractGradient,
)
from pydrake.systems.controllers import LinearQuadraticRegulator

import compatible_clf_cbf.clf as clf
from examples.quadrotor.plant import QuadrotorPlant


def lqr(quadrotor: QuadrotorPlant) -> Tuple[np.ndarray, np.ndarray]:
    x_equilibrium = np.zeros((12,))
    u_equilibrium = np.full((4,), quadrotor.m * quadrotor.g / 4)
    xu = np.concatenate((x_equilibrium, u_equilibrium))
    xu_ad = InitializeAutoDiff(xu).reshape((-1,))
    xdot_ad = quadrotor.dynamics(xu_ad[:12], xu_ad[-4:], T=AutoDiffXd)
    xdot_grad = ExtractGradient(xdot_ad)
    A = xdot_grad[:, :12]
    B = xdot_grad[:, -4:]
    K, S = LinearQuadraticRegulator(A, B, Q=np.eye(12), R=np.eye(4))
    return K, S


def search_clf():
    x = sym.MakeVectorContinuousVariable(12, "x")
    quadrotor = QuadrotorPlant()
    f, g = quadrotor.affine_dynamics_taylor(x, np.zeros((12,)), f_degree=3, g_degree=2)

    u_bound = quadrotor.m * quadrotor.g
    u_vertices = np.array(list(itertools.product([0, u_bound], repeat=4)))

    clf_search = clf.ControlLyapunov(
        f=f,
        g=g,
        x=x,
        x_equilibrium=np.zeros((12,)),
        u_vertices=u_vertices,
        state_eq_constraints=None,
    )
    lagrangian_degrees = clf.ClfWInputLimitLagrangianDegrees(
        V_minus_rho=2, Vdot=[2] * 16, state_eq_constraints=None
    )

    _, lqr_S = lqr(quadrotor)
    V_init = sym.Polynomial(x.dot(lqr_S @ x))

    kappa = 0.1
    clf_degree = 2
    x_equilibrium = np.zeros((12,))

    candidate_stable_states = np.zeros((4, 12))
    candidate_stable_states[0, :3] = np.array([-1.5, 0, 0])
    candidate_stable_states[1, :3] = np.array([1.5, 0, 0])
    candidate_stable_states[2, :3] = np.array([0, 1.5, 0])
    candidate_stable_states[3, :3] = np.array([0, -1.5, 0])

    stable_states_options = clf.StableStatesOptions(
        candidate_stable_states, V_margin=0.01
    )

    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 1)

    V = clf_search.bilinear_alternation(
        V_init,
        lagrangian_degrees,
        kappa,
        clf_degree,
        x_equilibrium,
        max_iter=5,
        stable_states_options=stable_states_options,
        solver_options=solver_options,
    )
    x_set = sym.Variables(x)
    clf.save_clf(
        V,
        x_set,
        kappa,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/quadrotor_taylor_clf.pkl",
        ),
    )


if __name__ == "__main__":
    search_clf()
