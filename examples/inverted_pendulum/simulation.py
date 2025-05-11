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
from examples.inverted_pendulum.plant import InvertedPendulumPlant

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
    theta_range: List[float], theta_dot_range: List[float], num_samples: int
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
    assert len(theta_range) == 2
    assert len(theta_dot_range) == 2
    assert num_samples > 0
    grid_theta, grid_theta_dot = np.meshgrid(
        np.linspace(theta_range[0], theta_range[1], num_samples),
        np.linspace(theta_dot_range[0], theta_dot_range[1], num_samples),
    )
    grid_x = np.concatenate(
        (
            np.sin(grid_theta.reshape((1, -1))),
            np.cos(grid_theta.reshape((1, -1))) - 1,
            grid_theta_dot.reshape((1, -1)),
        ),
        axis=0,
    )
    return grid_theta, grid_theta_dot, grid_x


def plot_clf_cbf(
    ax: matplotlib.axes.Axes,
    V: sym.Polynomial,
    h: np.ndarray,
    x: np.ndarray,
    fill_compatible: bool,
) -> Tuple[
    matplotlib.contour.QuadContourSet,
    matplotlib.contour.QuadContourSet,
    Optional[matplotlib.contour.QuadContourSet],
]:
    """
    Plot the CLF/CBF in the θ, θdot plane. x refers to the extended state
    after polynomialization.
    """
    (
        grid_theta,
        grid_theta_dot,
        grid_x,
    ) = sample_state_space(
        theta_range=[(3 / 4) * np.pi, -(3 / 4) * np.pi],
        theta_dot_range=[0.8, -0.8],
        num_samples=300,
    )
    grid_V = V.EvaluateIndeterminates(x, grid_x).reshape(grid_theta.shape)
    grid_h = h[0].EvaluateIndeterminates(x, grid_x).reshape(grid_theta.shape)
    h_V = ax.contour(
        grid_theta, grid_theta_dot, grid_V, levels=np.array([1]), colors="red"
    )
    h_h = ax.contour(
        grid_theta, grid_theta_dot, grid_h, levels=np.array([0]), colors="blue"
    )

    if fill_compatible:
        # Fill in the region {x|V(x)<=1, h(x) >= 0}, namely
        # {x | max(V(x)-1, -h(x)) <= 0}.
        grid_fill_vals = np.maximum(grid_V - 1, -grid_h)
        h_compatible = ax.contourf(
            grid_theta,
            grid_theta_dot,
            grid_fill_vals,
            levels=[-np.inf, 0],
            colors="green",
            alpha=0.4,
        )
    else:
        h_compatible = None

    return h_V, h_h, h_compatible


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
        grid_theta,
        grid_theta_dot,
        grid_x,
    ) = sample_state_space(
        theta_range=[(3 / 4) * np.pi, -(3 / 4) * np.pi],
        theta_dot_range=[0.8, -0.8],
        num_samples=100,
    )
    grid_unsafe = np.array(
        [
            unsafe_region[i].EvaluateIndeterminates(x, grid_x).reshape(grid_theta.shape)
            for i in range(len(unsafe_region))
        ]
    )
    unsafe_fill_vals = np.max(grid_unsafe, axis=0)
    unsafe_region_filled = ax.contourf(
        grid_theta,
        grid_theta_dot,
        unsafe_fill_vals,
        levels=[0, np.inf],
        colors="black",
        alpha=0.4,
    )
    return unsafe_region_filled


def synthesis_result_visualization():
    """
    This function generates the Fig.2 of the journal paper
    """
    # load the data
    filename_init = "inverted_pendulum_clf_cbf_init.pkl"
    filename_synth = "inverted_pendulum_clf_cbf.pkl"
    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)
    V_init, h_init, V, h, _, _, _ = load_data(filename_init, filename_synth, x_set)

    # plot the synthesis results:
    fig = plt.figure()
    ax = fig.add_subplot()
    # plot the initial CLF and CBF(contours)
    h_V_init, h_h_init, _ = plot_clf_cbf(ax, V_init, h_init, x, fill_compatible=False)
    h_V_init.set(linestyle="dashed")
    h_h_init.set(linestyle="dashed")
    # plot the synthesized CLF and CBF(compatible region filled)
    h_V, h_h, h_compatible = plot_clf_cbf(ax, V, h, x, fill_compatible=True)
    # plot the unsafe region
    unsafe_region = np.array(
        [-sym.Polynomial(0 * x[0] + x[1] + (1 - np.cos(np.pi / 2)))]
    )
    unsafe_region = plot_unsafe_region(ax, unsafe_region, x)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_xticks(
        [
            -(3 / 4) * np.pi,
            -(1 / 2) * np.pi,
            -(1 / 4) * np.pi,
            0,
            (1 / 4) * np.pi,
            (1 / 2) * np.pi,
            (3 / 4) * np.pi,
        ]
    )
    ax.set_xticklabels(
        [
            r"$-\frac{3}{4}\pi$",
            r"$-\frac{1}{2}\pi$",
            r"$-\frac{1}{4}\pi$",
            r"$0$",
            r"$\frac{1}{4}\pi$",
            r"$\frac{1}{2}\pi$",
            r"$\frac{3}{4}\pi$",
        ],
        fontsize=14,
    )
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_yticklabels([-0.8, -0.4, 0, 0.4, 0.8], fontsize=12)
    ax.set_xlim([-(3 / 4) * np.pi, (3 / 4) * np.pi])
    ax.set_ylim([-0.8, 0.8])
    ax.legend(
        [
            h_V_init.legend_elements()[0][0],
            h_h_init.legend_elements()[0][0],
            h_V.legend_elements()[0][0],
            h_h.legend_elements()[0][0],
            h_compatible.legend_elements()[0][0],
            unsafe_region.legend_elements()[0][0],
        ],
        [
            r"$V_{init}(x) = 1$",
            r"$h_{init}(x) = 0$",
            r"$V(x) = 1$",
            r"$h(x) = 0$",
            r"Compatible Region",
            r"Unsafe region",
        ],
        loc="upper right",
        prop={"size": 12},
    )
    fig.show()


def build_diagram() -> Tuple[
    Diagram,
    InvertedPendulumPlant,
    ClfCbfController,
    VectorLogSink,
    VectorLogSink,
    VectorLogSink,
    VectorLogSink,
]:
    builder = DiagramBuilder()
    inverted_pendulum = builder.AddSystem(InvertedPendulumPlant())
    state_logger = LogVectorOutput(inverted_pendulum.get_output_port(), builder)
    poly_plant = InvertedPendulumPlant()
    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)

    f, g = poly_plant.trig_poly_affine_dynamics(x)
    _, _, V, h, kappaV, kappah, _ = load_data(
        filename_init="inverted_pendulum_clf_cbf_init.pkl",
        filename_synth="inverted_pendulum_clf_cbf.pkl",
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
        clf_cbf_controller.action_output_port(), inverted_pendulum.get_input_port(0)
    )
    builder.Connect(
        inverted_pendulum.get_output_port(0), clf_cbf_controller.get_input_port(0)
    )

    action_logger = LogVectorOutput(clf_cbf_controller.action_output_port(), builder)

    clf_logger = LogVectorOutput(clf_cbf_controller.clf_output_port(), builder)

    cbf_logger = LogVectorOutput(clf_cbf_controller.cbf_output_port(), builder)
    diagram = builder.Build()
    return (
        diagram,
        inverted_pendulum,
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
        inverted_pendulum,
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
    This function simulates the inverted pendulum system
    with the CLF-CBF-QP controller. Then it plots the control
    input versus time, which is the plot shown in Fig.4(a)
    of the journal paper.
    """
    # set initial states. There are 25 initial states in total.
    theta = np.linspace(-np.pi / 3, np.pi / 3, 5)
    theta_dot = np.linspace(-0.3, 0.3, 5)
    theta, theta_dot = np.meshgrid(theta, theta_dot)
    initial_states = np.stack([theta, theta_dot], axis=-1)
    initial_states = initial_states.reshape(-1, 2)
    initial_states_extended = np.concatenate(
        (
            np.sin(initial_states[:, 0]).reshape(-1, 1),
            np.cos(initial_states[:, 0]).reshape(-1, 1) - 1,
            initial_states[:, 1].reshape(-1, 1),
        ),
        axis=1,
    )

    fig = plt.figure()
    ax = fig.add_subplot()
    # pick some of the initial states and plot the control input
    for i in range(1, 5):
        state_data, action_data, clf_data, cbf_data, time_data = simulate(
            initial_states_extended[i], duration=15
        )
        # plot the control versus time
        ax.plot(time_data, action_data[0])
    # plot the control input limits:
    x_axis = np.linspace(0, 15, 100)
    (control_limit_contour) = ax.plot(
        x_axis, 10 * np.ones_like(x_axis), linestyle="dashed", color="black"
    )
    ax.plot(x_axis, -10 * np.ones_like(x_axis), linestyle="dashed", color="black")
    # set plot properties
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control input")
    ax.set_title("Control input versus time")
    ax.set_xlim([0, 15])
    ax.set_ylim([-11, 11])
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels([0, 5, 10, 15], fontsize=12)
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_yticklabels([-10, -5, 0, 5, 10], fontsize=12)
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
    synthesis_result_visualization()  # generates Fig.2 of the journal paper
    run_simulations()  # generates Fig.4(a) of the journal paper


if __name__ == "__main__":
    main()
