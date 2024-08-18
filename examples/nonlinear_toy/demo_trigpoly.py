"""
Find the CLF/CBF without the input limits.
We use the trigonometric state with polynomial dynamics.
"""

import argparse
import os
from typing import Optional, Tuple

import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt
import numpy as np

import pydrake.symbolic as sym
import pydrake.solvers as solvers

import compatible_clf_cbf.clf_cbf as clf_cbf
import compatible_clf_cbf.utils as utils
import examples.nonlinear_toy.toy_system as toy_system


def plot_clf_cbf(
    ax: matplotlib.axes.Axes,
    V: sym.Polynomial,
    b: np.ndarray,
    x: np.ndarray,
    fill_compatible: bool,
) -> Tuple[
    matplotlib.contour.QuadContourSet,
    matplotlib.contour.QuadContourSet,
    Optional[matplotlib.contour.QuadContourSet],
]:
    """
    Plot the CLF/CBF in the θ, θdot plane.

    Args:
      fill_compatible: Fill in the compatible region.
    """
    grid_theta, grid_theta_dot = np.meshgrid(
        np.linspace(-np.pi, np.pi, 100), np.linspace(-3, 3, 100)
    )
    grid_x_vals = np.concatenate(
        (
            np.sin(grid_theta.reshape((1, -1))),
            np.cos(grid_theta.reshape((1, -1))) - 1,
            grid_theta_dot.reshape((1, -1)),
        ),
        axis=0,
    )
    grid_V = V.EvaluateIndeterminates(x, grid_x_vals).reshape(grid_theta.shape)
    grid_b = b[0].EvaluateIndeterminates(x, grid_x_vals).reshape(grid_theta.shape)
    h_V = ax.contour(
        grid_theta, grid_theta_dot, grid_V, levels=np.array([1]), colors="red"
    )
    h_b = ax.contour(
        grid_theta, grid_theta_dot, grid_b, levels=np.array([0]), colors="blue"
    )

    if fill_compatible:
        # Fill in the region {x|V(x)<=1, b(x) >= 0}, namely
        # {x | max(V(x)-1, -b(x)) <= 0}.
        grid_fill_vals = np.maximum(grid_V - 1, -grid_b)
        h_compatible = ax.contourf(
            grid_theta,
            grid_theta_dot,
            grid_fill_vals,
            levels=[-np.inf, 0],
            colors="green",
            alpha=0.2,
        )
    else:
        h_compatible = None

    return h_V, h_b, h_compatible


def get_unsafe_regions(x: np.ndarray) -> np.ndarray:
    return np.array([sym.Polynomial(x[0] + x[1] + x[2] + 2)])


def plot_unsafe_regions(ax: matplotlib.axes.Axes):
    x = sym.MakeVectorContinuousVariable(3, "x")
    unsafe_regions = get_unsafe_regions(x)
    grid_theta, grid_theta_dot = np.meshgrid(
        np.linspace(-np.pi, np.pi, 100), np.linspace(-3, 3, 100)
    )
    grid_x_vals = np.concatenate(
        (
            np.sin(grid_theta.reshape((1, -1))),
            np.cos(grid_theta.reshape((1, -1))) - 1,
            grid_theta_dot.reshape((1, -1)),
        ),
        axis=0,
    )
    unsafe_values = (
        np.concatenate(
            [
                region.EvaluateIndeterminates(x, grid_x_vals).reshape((1, -1))
                for region in unsafe_regions
            ],
            axis=0,
        )
        .max(axis=0)
        .reshape(grid_theta.shape)
    )
    h_unsafe = ax.contourf(
        grid_theta,
        grid_theta_dot,
        unsafe_values,
        levels=np.array([-np.inf, 0.0]),
        alpha=0.5,
        colors="grey",
    )
    return h_unsafe


def get_clf_cbf_init(x: np.ndarray) -> Tuple[sym.Polynomial, np.ndarray]:
    V_init = sym.Polynomial(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) / 0.1
    b_init = np.array([sym.Polynomial(0.1 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2)])
    return V_init, b_init


def plot_clf_cbf_init(
    ax: matplotlib.axes.Axes,
) -> Tuple[matplotlib.contour.QuadContourSet, matplotlib.contour.QuadContourSet]:
    x = sym.MakeVectorContinuousVariable(3, "x")
    V_init, b_init = get_clf_cbf_init(x)
    h_V_init, h_b_init, _ = plot_clf_cbf(ax, V_init, b_init, x, fill_compatible=False)
    h_V_init.set(linestyle="dotted", edgecolor="r")
    h_b_init.set(linestyle=(0, (3, 5, 1, 5)), edgecolor="b")
    return h_V_init, h_b_init


def search(unit_test_flag: bool = False):
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = toy_system.affine_trig_poly_dynamics(x)
    state_eq_constraints = np.array([toy_system.affine_trig_poly_state_constraints(x)])
    use_y_squared = True
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        unsafe_regions=[get_unsafe_regions(x)],
        Au=np.array([[1], [-1]]),
        bu=np.array([1, 1]),
        with_clf=True,
        use_y_squared=use_y_squared,
        state_eq_constraints=state_eq_constraints,
    )
    V_init, b_init = get_clf_cbf_init(x)

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
    max_iter = 20

    binary_search_scale_options = None
    inner_ellipsoid_options = None
    compatible_states_options = clf_cbf.CompatibleStatesOptions(
        candidate_compatible_states=np.array(
            [
                [np.sin(-np.pi / 3), np.cos(-np.pi / 3) - 1, -0.5],
                [np.sin(0), np.cos(0) - 1, -1.5],
                [np.sin(np.pi / 2), np.cos(np.pi / 2) - 1, -1.9],
            ]
        ),
        anchor_states=np.array([[0.0, 0, 0]]),
        b_anchor_bounds=[(np.array([0]), np.array([0.1]))],
        weight_V=1,
        weight_b=np.array([1.0]),
        b_margins=np.array([0.01]),
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
        inner_ellipsoid_options=inner_ellipsoid_options,
        binary_search_scale_options=binary_search_scale_options,
        compatible_states_options=compatible_states_options,
        solver_options=solver_options,
        backoff_scale=utils.BackoffScale(rel=None, abs=0.01),
    )
    print(f"V={V}")
    print(f"b={b}")
    assert V is not None
    x_set = sym.Variables(x)
    if not unit_test_flag:
        clf_cbf.save_clf_cbf(
            V,
            b,
            x_set,
            kappa_V,
            kappa_b,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../data/nonlinear_toy_clf_cbf.pkl",
            ),
        )
    return V, b


def visualize():
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = toy_system.affine_trig_poly_dynamics(x)
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data/nonlinear_toy_clf_cbf.pkl",
    )
    x_set = sym.Variables(x)
    saved_data = clf_cbf.load_clf_cbf(path, x_set)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel(r"$\theta$ (rad)", fontsize=16)
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)", fontsize=16)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"0", r"$\frac{\pi}{2}$", r"$\pi$"]
    )
    ax.tick_params(axis="both", which="major", labelsize=14)
    h_V, h_b, h_compatible = plot_clf_cbf(
        ax, saved_data["V"], saved_data["b"], x, fill_compatible=True
    )
    h_V_init, h_b_init = plot_clf_cbf_init(ax)
    plot_unsafe_regions(ax)
    ax.legend(
        [
            h_V.legend_elements()[0][0],
            h_b.legend_elements()[0][0],
            h_V_init.legend_elements()[0][0],
            h_b_init.legend_elements()[0][0],
        ],
        [r"$V(x)=1$", r"$b(x)=0$", r"$V_{init}(x)=1$", r"$b_{init}(x)=0$"],
        prop={"size": 12},
    )
    fig.show()
    for fig_extension in (".png", "pdf"):
        fig.savefig(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"../../figures/nonlinear_toy{fig_extension}",
            ),
            bbox_inches="tight",
        )

    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unit_test", action="store_true", help="Only turn this on in the unit test."
    )
    args = parser.parse_args()
    V, b = search(args.unit_test)
    if not args.unit_test:
        visualize()


if __name__ == "__main__":
    main()
