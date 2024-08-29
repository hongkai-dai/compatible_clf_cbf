"""
Certify the compatible CLF/CBF without input limits.
This uses the polynomial dynamics of the quadrotor (with quaternion as part of
the state), and equality constraint to enforce the unit-length constraint on the
quaternion.
"""

import os
from typing import Tuple

import numpy as np

from pydrake.geometry import (
    Box,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    Rgba,
    StartMeshcat,
    SceneGraph,
)
import pydrake.math
import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.analysis
from pydrake.systems.framework import Diagram, DiagramBuilder
import pydrake.systems.framework
from pydrake.systems.primitives import LogVectorOutput, VectorLogSink


import compatible_clf_cbf.clf as clf
from compatible_clf_cbf import clf_cbf
from examples.quadrotor.plant import QuadrotorPolyPlant, QuadrotorPolyGeometry
import examples.quadrotor.demo_clf
from compatible_clf_cbf.utils import BackoffScale
import compatible_clf_cbf.controller


def search(use_y_squared: bool, with_u_bound: bool):
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
                "../../data/quadrotor_clf_cbf8.pkl",
            ),
            x_set,
        )
        V_init = data["V"]
        b_init = data["b"]
    kappa_V_sequences = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.2, 0.25, 0.625]
    kappa_b_sequences = [np.array([0.2]) for _ in range(len(kappa_V_sequences))]

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

    candidate_compatible_states_sequences.append(np.zeros((5, 13)))
    candidate_compatible_states_sequences[-1][1, 5] = -0.6
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = 0.6
    candidate_compatible_states_sequences[-1][2, 6] = -0.35
    candidate_compatible_states_sequences[-1][3, 4] = 0.6
    candidate_compatible_states_sequences[-1][3, 6] = -0.35
    candidate_compatible_states_sequences[-1][4, 4] = -0.6
    candidate_compatible_states_sequences[-1][4, 6] = -0.35

    candidate_compatible_states_sequences.append(np.zeros((5, 13)))
    candidate_compatible_states_sequences[-1][1, 5] = -0.7
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = 0.7
    candidate_compatible_states_sequences[-1][2, 6] = -0.35
    candidate_compatible_states_sequences[-1][3, 4] = 0.7
    candidate_compatible_states_sequences[-1][3, 6] = -0.35
    candidate_compatible_states_sequences[-1][4, 4] = -0.7
    candidate_compatible_states_sequences[-1][4, 6] = -0.35

    candidate_compatible_states_sequences.append(np.zeros((5, 13)))
    candidate_compatible_states_sequences[-1][1, 5] = -0.8
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = 0.8
    candidate_compatible_states_sequences[-1][2, 6] = -0.35
    candidate_compatible_states_sequences[-1][3, 4] = 0.8
    candidate_compatible_states_sequences[-1][3, 6] = -0.35
    candidate_compatible_states_sequences[-1][4, 4] = -0.8
    candidate_compatible_states_sequences[-1][4, 6] = -0.35

    candidate_compatible_states_sequences.append(np.zeros((5, 13)))
    candidate_compatible_states_sequences[-1][1, 5] = -0.9
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = 0.9
    candidate_compatible_states_sequences[-1][2, 6] = -0.35
    candidate_compatible_states_sequences[-1][3, 4] = 0.9
    candidate_compatible_states_sequences[-1][3, 6] = -0.35
    candidate_compatible_states_sequences[-1][4, 4] = -0.9
    candidate_compatible_states_sequences[-1][4, 6] = -0.35

    candidate_compatible_states_sequences.append(np.zeros((5, 13)))
    candidate_compatible_states_sequences[-1][1, 5] = -1
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = 1
    candidate_compatible_states_sequences[-1][2, 6] = -0.35
    candidate_compatible_states_sequences[-1][3, 4] = 1
    candidate_compatible_states_sequences[-1][3, 6] = -0.35
    candidate_compatible_states_sequences[-1][4, 4] = -1
    candidate_compatible_states_sequences[-1][4, 6] = -0.35

    candidate_compatible_states_sequences.append(np.zeros((7, 13)))
    candidate_compatible_states_sequences[-1][1, 5] = -1
    candidate_compatible_states_sequences[-1][1, 6] = -0.35
    candidate_compatible_states_sequences[-1][2, 5] = 1
    candidate_compatible_states_sequences[-1][2, 6] = -0.35
    candidate_compatible_states_sequences[-1][3, 4] = 1
    candidate_compatible_states_sequences[-1][3, 6] = -0.35
    candidate_compatible_states_sequences[-1][4, 4] = -1
    candidate_compatible_states_sequences[-1][4, 6] = -0.35
    candidate_compatible_states_sequences[-1][5, 5] = -1
    candidate_compatible_states_sequences[-1][5, :4] = np.array(
        [np.cos(np.pi / 36) - 1, np.sin(np.pi / 36), 0, 0]
    )
    candidate_compatible_states_sequences[-1][6, 5] = 1
    candidate_compatible_states_sequences[-1][6, :4] = np.array(
        [np.cos(np.pi / 18) - 1, np.sin(np.pi / 18), 0, 0]
    )

    candidate_compatible_states_sequences.append(np.zeros((7, 13)))
    candidate_compatible_states_sequences[-1] = candidate_compatible_states_sequences[
        -2
    ]

    lagrangian_sos_types = [
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
    ]
    V_margin_sequence = [None, None, None, None, None, 0.01, 0.01, 0.01, 0.05, 0.05]
    b_margins_sequence = [
        None,
        None,
        np.array([0.05]),
        np.array([0.04]),
        np.array([0.02]),
        np.array([0.02]),
        np.array([0.03]),
        np.array([0.03]),
        np.array([0.02]),
        np.array([0.02]),
    ]
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, True)
    V = V_init
    b = b_init
    max_iter_sequence = [1, 1, 1, 2, 1, 1, 2, 2, 3, 1, 1]
    backoff_scale_sequence = [
        BackoffScale(rel=None, abs=0.2),
        BackoffScale(rel=None, abs=0.2),
        BackoffScale(rel=None, abs=0.2),
        BackoffScale(rel=None, abs=0.1),
        BackoffScale(rel=None, abs=0.2),
        BackoffScale(rel=None, abs=0.1),
        BackoffScale(rel=None, abs=0.1),
        BackoffScale(rel=None, abs=0.15),
        BackoffScale(rel=None, abs=0.1),
        BackoffScale(rel=None, abs=0.1),
    ]
    for i in range(9, 10):  # len(candidate_compatible_states_sequences)):
        compatible_states_options = clf_cbf.CompatibleStatesOptions(
            candidate_compatible_states=candidate_compatible_states_sequences[i],
            anchor_states=np.zeros((1, 13)),
            b_anchor_bounds=[(np.array([0.6]), np.array([1.0]))],
            weight_V=1,
            weight_b=np.array([1]),
            V_margin=V_margin_sequence[i],
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


def build_diagram() -> Tuple[
    Diagram,
    QuadrotorPolyPlant,
    compatible_clf_cbf.controller.ClfCbfController,
    VectorLogSink,
    VectorLogSink,
    VectorLogSink,
    VectorLogSink,
]:
    builder = DiagramBuilder()
    quadrotor = builder.AddSystem(QuadrotorPolyPlant())
    scene_graph = builder.AddSystem(SceneGraph())
    QuadrotorPolyGeometry.AddToBuilder(
        builder, quadrotor.get_output_port(0), "quadrotor", scene_graph
    )

    meshcat = StartMeshcat()

    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, MeshcatVisualizerParams(role=Role.kPerception)
    )
    meshcat.SetObject("ground", Box(10, 10, 0.1), rgba=Rgba(0.5, 0.5, 0.5))
    meshcat.SetTransform("ground", pydrake.math.RigidTransform(np.array([0, 0, -0.55])))

    state_logger = LogVectorOutput(quadrotor.get_output_port(), builder)

    poly_plant = QuadrotorPolyPlant()
    x = sym.MakeVectorContinuousVariable(13, "x")
    x_set = sym.Variables(x)

    f, g = poly_plant.affine_dynamics(x)

    pickle_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data/quadrotor_clf_cbf9.pkl",
    )
    clf_cbf_data = clf_cbf.load_clf_cbf(pickle_path, x_set)
    V = clf_cbf_data["V"]
    b = clf_cbf_data["b"]
    kappa_V = clf_cbf_data["kappa_V"]
    kappa_b = clf_cbf_data["kappa_b"]

    Qu = np.eye(4)
    clf_cbf_controller = builder.AddSystem(
        compatible_clf_cbf.controller.ClfCbfController(
            f,
            g,
            V,
            b,
            x,
            kappa_V,
            kappa_b,
            Qu,
            Au=None,
            bu=None,
            solver_id=None,
            solver_options=None,
        )
    )
    builder.Connect(
        clf_cbf_controller.action_output_port(), quadrotor.get_input_port(0)
    )
    builder.Connect(quadrotor.get_output_port(0), clf_cbf_controller.get_input_port(0))

    action_logger = LogVectorOutput(clf_cbf_controller.action_output_port(), builder)

    clf_logger = LogVectorOutput(clf_cbf_controller.clf_output_port(), builder)

    cbf_logger = LogVectorOutput(clf_cbf_controller.cbf_output_port(), builder)
    diagram = builder.Build()
    return (
        diagram,
        quadrotor,
        clf_cbf_controller,
        state_logger,
        action_logger,
        clf_logger,
        cbf_logger,
    )


def simulate(x0: np.ndarray, duration: float):
    (
        diagram,
        quadrotor,
        clf_cbf_controller,
        state_logger,
        action_logger,
        clf_logger,
        cbf_logger,
    ) = build_diagram()
    simulator = pydrake.systems.analysis.Simulator(diagram)
    simulator.get_mutable_context().SetContinuousState(x0)

    def monitor(context):
        quadrotor_context = quadrotor.GetMyContextFromRoot(context)
        x_val = quadrotor.get_output_port(0).Eval(quadrotor_context)
        if np.linalg.norm(x_val) < 1e-2:
            return pydrake.systems.framework.EventStatus.ReachedTermination(
                diagram, "reach_goal"
            )
        return pydrake.systems.framework.EventStatus.Succeeded()

    simulator.set_monitor(monitor)
    simulator_config = pydrake.systems.analysis.SimulatorConfig(
        integration_scheme="runge_kutta3"
    )
    pydrake.systems.analysis.ApplySimulatorConfig(simulator_config, simulator)
    simulator.AdvanceTo(duration)

    state_data = state_logger.FindLog(simulator.get_context()).data()
    action_data = action_logger.FindLog(simulator.get_context()).data()
    clf_data = clf_logger.FindLog(simulator.get_context()).data()
    cbf_data = cbf_logger.FindLog(simulator.get_context()).data()
    time_data = state_logger.FindLog(simulator.get_context()).sample_times()
    return state_data, action_data, clf_data, cbf_data, time_data


def run_simulations():
    x0_sequences = []
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[0][4:7] = np.array([1, 0, -0.3])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[1][4:7] = np.array([-1, 0, -0.3])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[2][4:7] = np.array([0, 1, -0.3])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[3][4:7] = np.array([0, -1, -0.3])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[4][4:7] = np.array([0.2, -0.8, 0.1])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[5][4:7] = np.array([0.1, -0.95, 0])
    x0_sequences[5][:4] = np.array([np.cos(np.pi / 36) - 1, np.sin(np.pi / 36), 0, 0])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[6][4:7] = np.array([0.5, 0.7, 0.1])
    x0_sequences[6][:4] = np.array([np.cos(np.pi / 20) - 1, np.sin(np.pi / 20), 0, 0])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[7][4:7] = np.array([-0.4, 0.75, -0.1])
    x0_sequences[7][:4] = np.array([np.cos(np.pi / 15) - 1, np.sin(np.pi / 15), 0, 0])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[8][4:7] = np.array([-0.6, -0.55, -0.2])
    x0_sequences[8][:4] = np.array([np.cos(np.pi / 18) - 1, 0, np.sin(np.pi / 18), 0])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[9][4:7] = np.array([0.6, -0.15, 0.05])
    x0_sequences[9][:4] = np.array([np.cos(np.pi / 15) - 1, 0, np.sin(-np.pi / 15), 0])
    x0_sequences.append(np.zeros((13,)))
    x0_sequences[10][4:7] = np.array([-0.8, -0.05, -0.4])
    x0_sequences[10][:4] = np.array([np.cos(np.pi / 30) - 1, 0, np.sin(np.pi / 30), 0])
    for i in range(len(x0_sequences)):
        state_data, action_data, clf_data, cbf_data, time_data = simulate(
            x0_sequences[i], duration=60
        )
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../../data/quadrotor_sim{i}.npz",
        )
        np.savez(
            path,
            state_data=state_data,
            action_data=action_data,
            clf_data=clf_data,
            cbf_data=cbf_data,
            time_data=time_data,
        )


def main():
    search(use_y_squared=True, with_u_bound=False)
    # run_simulations()


if __name__ == "__main__":
    with solvers.MosekSolver.AcquireLicense():
        main()
