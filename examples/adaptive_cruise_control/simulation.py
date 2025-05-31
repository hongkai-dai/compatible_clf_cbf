import os
import numpy as np
from typing import Optional, Tuple, List, Union

import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt

import pydrake.symbolic as sym
import pydrake.systems.analysis
from pydrake.systems.framework import Diagram, DiagramBuilder
import pydrake.systems.framework
from pydrake.systems.primitives import LogVectorOutput, VectorLogSink
from examples.adaptive_cruise_control.plant import AdaptiveCruiseControlPlant

import compatible_clf_cbf.clf_cbf as clf_cbf
from compatible_clf_cbf.controller import ClfCbfController


def load_data(filename_init: str, filename_synth: str, x_set: sym.Variable) -> Tuple[
    sym.Polynomial,  # the initial CLF
    np.ndarray,  # the initial CBF
    sym.Polynomial,  # the synthesized CLF
    np.ndarray,  # the synthesized CBF
    float,  # kappaV
    Union[np.ndarray, List[List[float]]],  # kappa_h
    List,  # relative degree
]:
    path_init = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename_init
    )
    path_synth = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename_synth
    )
    loaded_data_init = clf_cbf.load_clf_cbf(path_init, x_set)
    loaded_data_synth = clf_cbf.load_clf_cbf(path_synth, x_set)
    V_init = loaded_data_init["V"]
    h_init = loaded_data_init["h"]
    V = loaded_data_synth["V"]
    h = loaded_data_synth["h"]
    kappaV = loaded_data_synth["kappa_V"]
    kappah = loaded_data_synth["kappa_h"]
    if isinstance(kappah, List):
        relative_degree = [len(kappah[i]) for i in range(len(kappah))]
    else:
        relative_degree = None
    return (V_init, h_init, V, h, kappaV, kappah, relative_degree)


def sample_state_space(
    velocity_range: List[float], position_range: List[float], num_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the original state space of the inverted pendulum.
    Args:
        theta_range: the range of theta.
        theta_dot_range: the range of theta dot.
        num_samples: the number of samples.
    Returns:
        grid_theta: the sampled theta values.
        grid_theta_dot: the sampled theta dot values.
        grid_x: the sampled states in the extended state space.
    """
    assert len(velocity_range) == 2
    assert len(position_range) == 2
    assert num_samples > 0
    grid_velocity, grid_position = np.meshgrid(
        np.linspace(velocity_range[0], velocity_range[1], num_samples),
        np.linspace(position_range[0], position_range[1], num_samples),
    )
    grid_x = np.concatenate(
        (grid_velocity.reshape((1, -1)), grid_position.reshape((1, -1))),
        axis=0,
    )
    return grid_velocity, grid_position, grid_x


def plot_clf_cbf(
    ax: matplotlib.axes.Axes,
    V: sym.Polynomial,
    h: np.ndarray,
    x: np.ndarray,
    fill_compatible: bool,
) -> Tuple[
    matplotlib.contour.QuadContourSet,
    matplotlib.contour.QuadContourSet,
    matplotlib.contour.QuadContourSet,
    Optional[matplotlib.contour.QuadContourSet],
]:
    """
    Plot the CLF/CBF in the θ, θdot plane. x refers to the extended state
    after polynomialization.
    """
    (
        grid_velocity,
        grid_position,
        grid_x,
    ) = sample_state_space(
        velocity_range=[-1.5, 1.5],
        position_range=[-3, 3],
        num_samples=300,
    )
    grid_V = V.EvaluateIndeterminates(x, grid_x).reshape(grid_velocity.shape)
    grid_h = np.zeros(
        shape=(h.shape[0], grid_velocity.shape[0], grid_velocity.shape[1])
    )
    for i in range(h.shape[0]):
        grid_h[i] = h[i].EvaluateIndeterminates(x, grid_x).reshape(grid_velocity.shape)
    h_V = ax.contour(
        grid_position, grid_velocity, grid_V, levels=np.array([1]), colors="red"
    )
    h_h0 = ax.contour(
        grid_position, grid_velocity, grid_h[0], levels=np.array([0]), colors="blue"
    )
    h_h1 = ax.contour(
        grid_position, grid_velocity, grid_h[1], levels=np.array([0]), colors="purple"
    )
    if fill_compatible:
        # Fill in the region {x|V(x)<=1, h0(x) >= 0, h1(x) >= 0}, namely
        # {x | max(V(x)-1, -h0(x), -h1(x)) <= 0}.
        grid_V_extend = grid_V.reshape(1, grid_V.shape[0], grid_V.shape[1])
        grid_fill_vals = np.max(
            np.concatenate([grid_V_extend - 1, -grid_h], axis=0),
            axis=0,
        )
        h_compatible = ax.contourf(
            grid_position,
            grid_velocity,
            grid_fill_vals,
            levels=[-np.inf, 0],
            colors="green",
            alpha=0.4,
        )
    else:
        h_compatible = None

    return h_V, h_h0, h_h1, h_compatible


def plot_unsafe_region(
    ax: matplotlib.axes.Axes,
    unsafe_region: np.ndarray,
    x: np.ndarray,
) -> matplotlib.contour.QuadContourSet:
    """
    plot the unsafe region.
    Args:
        unsafe_region: an array of symbolic polynomials.
         defining the unsafe region as p_i(x)>= 0. i=1,2,...,n
        x: the extended state after polynomialization.
    """
    (
        grid_velocity,
        grid_position,
        grid_x,
    ) = sample_state_space(
        velocity_range=[-1.5, 1.5],
        position_range=[-3, 3],
        num_samples=300,
    )
    grid_unsafe = np.array(
        [
            unsafe_region[i]
            .EvaluateIndeterminates(x, grid_x)
            .reshape(grid_velocity.shape)
            for i in range(len(unsafe_region))
        ]
    )
    unsafe_fill_vals = np.max(grid_unsafe, axis=0)
    unsafe_region_filled = ax.contourf(
        grid_position,
        grid_velocity,
        unsafe_fill_vals,
        levels=[0, np.inf],
        colors="black",
        alpha=0.4,
    )
    return unsafe_region_filled


def synthesis_result_visualization():
    """
    This function generates the Fig.3 of the journal paper
    """
    # load the data
    filename_init = "acc_clf_cbf_init.pkl"
    filename_synth = "acc_clf_cbf.pkl"
    x = sym.MakeVectorContinuousVariable(2, "x")
    x_set = sym.Variables(x)
    V_init, h_init, V, h, _, _, _ = load_data(filename_init, filename_synth, x_set)

    # plot the synthesis results:
    fig = plt.figure()
    ax = fig.add_subplot()
    # plot the initial CLF and CBF(contours)
    (h_V_init, h_h_init0, h_h_init1, _) = plot_clf_cbf(
        ax, V_init, h_init, x, fill_compatible=False
    )
    h_V_init.set(linestyle="dashed")
    h_h_init0.set(linestyle="dashed")
    h_h_init1.set(linestyle="dashed")
    # plot the synthesized CLF and CBF(compatible region filled)
    h_V, h_h0, h_h1, h_compatible = plot_clf_cbf(ax, V, h, x, fill_compatible=True)
    # plot the unsafe region
    cruise_control = AdaptiveCruiseControlPlant()
    unsafe_region = np.array(
        [
            sym.Polynomial(x[1] - (cruise_control.d_max - cruise_control.d_ref)),
            sym.Polynomial(-x[1] + (cruise_control.d_min - cruise_control.d_ref)),
        ]
    )
    unsafe_region = plot_unsafe_region(ax, unsafe_region, x)
    ax.set_xlabel(r"Position $m$ ($x_2$)")
    ax.set_ylabel(r"Velocity $m/s$ ($x_1$)")
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(
        [
            r"-3",
            r"-2",
            r"-1",
            r"0",
            r"1",
            r"2",
            r"3",
        ],
        fontsize=14,
    )
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5])
    ax.set_yticklabels([-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5], fontsize=12)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-1.5, 1.5])
    ax.legend(
        [
            h_V_init.legend_elements()[0][0],
            h_h_init0.legend_elements()[0][0],
            h_h_init1.legend_elements()[0][0],
            h_V.legend_elements()[0][0],
            h_h0.legend_elements()[0][0],
            h_h1.legend_elements()[0][0],
            h_compatible.legend_elements()[0][0],
            unsafe_region.legend_elements()[0][0],
        ],
        [
            r"$V_{init}(x) = 1$",
            r"$h1_{init}(x) = 0$",
            r"$h2_{init}(x) = 0$",
            r"$V(x) = 1$",
            r"$h1(x) = 0$",
            r"$h2(x) = 0$",
            r"Compatible Region",
            r"Unsafe region",
        ],
        loc="upper right",
        prop={"size": 12},
    )
    fig.show()


def build_diagram() -> Tuple[
    Diagram,
    AdaptiveCruiseControlPlant,
    ClfCbfController,
    VectorLogSink,
    VectorLogSink,
    VectorLogSink,
    VectorLogSink,
]:
    builder = DiagramBuilder()
    acc = builder.AddSystem(AdaptiveCruiseControlPlant())
    state_logger = LogVectorOutput(acc.get_output_port(), builder)
    poly_plant = AdaptiveCruiseControlPlant()
    x = sym.MakeVectorContinuousVariable(2, "x")
    x_set = sym.Variables(x)

    f, g = poly_plant.affine_dynamics(x)
    _, _, V, h, kappaV, kappah, _ = load_data(
        filename_init="acc_clf_cbf_init.pkl",
        filename_synth="acc_clf_cbf.pkl",
        x_set=x_set,
    )
    Qu = np.eye(1)
    clf_cbf_controller = builder.AddSystem(
        ClfCbfController(
            f,
            g,
            V,
            h,
            x,
            kappaV,
            kappah,
            Qu,
            Au=None,
            bu=None,
            solver_id=None,
            solver_options=None,
        )
    )
    builder.Connect(
        clf_cbf_controller.action_output_port(), acc.get_input_port(0)
    )
    builder.Connect(
        acc.get_output_port(0), clf_cbf_controller.get_input_port(0)
    )

    action_logger = LogVectorOutput(clf_cbf_controller.action_output_port(), builder)

    clf_logger = LogVectorOutput(clf_cbf_controller.clf_output_port(), builder)

    cbf_logger = LogVectorOutput(clf_cbf_controller.cbf_output_port(), builder)
    diagram = builder.Build()
    return (
        diagram,
        acc,
        clf_cbf_controller,
        state_logger,
        action_logger,
        clf_logger,
        cbf_logger,
    )


def simulate(x0: np.ndarray, duration: float):
    # initialize the block diagram
    (
        diagram,
        acc,
        clf_cbf_controller,
        state_logger,
        action_logger,
        clf_logger,
        cbf_logger,
    ) = build_diagram()

    # set the initial state
    simulator = pydrake.systems.analysis.Simulator(diagram)
    simulator.get_mutable_context().SetContinuousState(x0)

    # configure the simulator
    simulator_config = pydrake.systems.analysis.SimulatorConfig(
        integration_scheme="runge_kutta3"
    )
    pydrake.systems.analysis.ApplySimulatorConfig(simulator_config, simulator)

    # run the simulation
    simulator.AdvanceTo(duration)

    # collect the data
    state_data = state_logger.FindLog(simulator.get_context()).data()
    action_data = action_logger.FindLog(simulator.get_context()).data()
    clf_data = clf_logger.FindLog(simulator.get_context()).data()
    cbf_data = cbf_logger.FindLog(simulator.get_context()).data()
    time_data = state_logger.FindLog(simulator.get_context()).sample_times()
    return state_data, action_data, clf_data, cbf_data, time_data


def run_simulations():
    """
    This function simulates the adaptive cruise control system
    with the CLF-CBF-QP controller. Then it plots the control
    input versus time, which is the plot shown in Fig.4(b)
    of the journal paper.
    """
    # set initial states. There are 36 initial states in total.
    initial_states = np.array(
            [[0, -1.25], [0, 1.25], [0.5, -1.0], [-1, 1.5], [-1, 1.8], [0.5, -1.5]]
        ).reshape(6, 2, 1)

    fig = plt.figure()
    ax = fig.add_subplot()
    # pick some of the initial states and plot the control input
    for i in range(0, 6):
        state_data, action_data, clf_data, cbf_data, time_data = simulate(
            initial_states[i], duration=10
        )
        # plot the control versus time
        ax.plot(time_data, action_data[0])
    # plot the control input limits:
    x_axis = np.linspace(0, 10, 100)
    (control_limit_contour) = ax.plot(
        x_axis, 3 * np.ones_like(x_axis), linestyle="dashed", color="black"
    )
    ax.plot(x_axis, -4 * np.ones_like(x_axis), linestyle="dashed", color="black")
    # set plot properties
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control input")
    ax.set_title("Control input versus time")
    ax.set_xlim([0, 10])
    ax.set_ylim([-5, 5])
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([0, 5, 10], fontsize=12)
    ax.set_yticks([-5, 0, 5])
    ax.set_yticklabels([-5, 0, 5], fontsize=12)
    ax.legend(
        [control_limit_contour[0]],
        [
            r"control input limits",
        ],
        loc="upper right",
        prop={"size": 12},
    )
    fig.show()


def main():
    synthesis_result_visualization()  # generates Fig.3 of the journal paper
    run_simulations()  # generates Fig.4(b) of the journal paper


if __name__ == "__main__":
    main()
