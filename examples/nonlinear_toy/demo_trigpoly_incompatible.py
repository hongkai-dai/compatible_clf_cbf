"""
Find the CLF/CBF without the input limits.
We use the trigonometric state with polynomial dynamics.
"""

import os

import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt
import numpy as np

import pydrake.symbolic as sym
import pydrake.solvers as solvers

import compatible_clf_cbf.clf_cbf as clf_cbf
import compatible_clf_cbf.clf as clf
import compatible_clf_cbf.utils as utils
import examples.nonlinear_toy.toy_system as toy_system
import examples.nonlinear_toy.demo_trigpoly as demo_trigpoly
import examples.nonlinear_toy.incompatibility


def get_clf_init(x: np.ndarray) -> sym.Polynomial:
    x_set = sym.Variables(x)
    clf_cbf_data = clf_cbf.load_clf_cbf(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/nonlinear_toy_clf_cbf.pkl",
        ),
        x_set,
    )
    V_init = clf_cbf_data["V"]
    return V_init


def get_cbf_init(x: np.ndarray):
    b_init = sym.Polynomial(x[0] + x[1] + x[2] + 2)
    return b_init


def search_clf(x: np.ndarray, kappa: float) -> sym.Polynomial:
    f, g = toy_system.affine_trig_poly_dynamics(x)
    state_eq_constraints = np.array([toy_system.affine_trig_poly_state_constraints(x)])
    V_init = get_clf_init(x) / 0.6
    # Now verify that the CLF is valid.
    clf_verifier = clf.ControlLyapunov(
        f=f,
        g=g,
        x=x,
        x_equilibrium=np.zeros((3,)),
        u_vertices=np.array([[1], [-1]]),
        state_eq_constraints=state_eq_constraints,
    )
    lagrangian_degrees = clf.ClfWInputLimitLagrangianDegrees(
        V_minus_rho=4, Vdot=[4, 4], state_eq_constraints=[4]
    )

    candidate_stable_states = np.array(
        [
            [np.sin(-0.9 * np.pi), np.cos(-0.9 * np.pi) - 1, 1],
            [np.sin(0.9 * np.pi), np.cos(0.9 * np.pi) - 1, 0],
        ]
    )
    stable_states_options = clf.StableStatesOptions(
        candidate_stable_states=candidate_stable_states, V_margin=0.01
    )
    backoff_scale = utils.BackoffScale(rel=None, abs=0.01)
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 1)
    V = clf_verifier.bilinear_alternation(
        V_init,
        lagrangian_degrees,
        kappa,
        clf_degree=2,
        x_equilibrium=np.zeros((3,)),
        max_iter=2,
        stable_states_options=stable_states_options,
        solver_options=solver_options,
        lagrangian_coefficient_tol=None,
        lagrangian_sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSdsos,
        backoff_scale=backoff_scale,
    )
    return V


def plot_error_region(
    ax: matplotlib.axes.Axes,
    V: sym.Polynomial,
    rho: float,
    b: sym.Polynomial,
    x: np.ndarray,
    kappa_V: float,
    kappa_b: float,
):
    """
    Plot the intersection of the incompatible region (where CLF and CBF don't
    have a common solution) and the region where V <= rho and b >= 0.
    """
    grid_theta, grid_x2, grid_x_vals = demo_trigpoly.get_grid_pts()
    compute_incompatible = (
        examples.nonlinear_toy.incompatibility.ComputeIncompatibility(
            V, b, x, kappa_V, kappa_b, u_min=-1, u_max=1
        )
    )
    grid_incompatible_vals = np.array(
        [
            compute_incompatible.eval(grid_x_vals[:, i])
            for i in range(grid_x_vals.shape[1])
        ]
    ).reshape(grid_theta.shape)

    grid_V = V.EvaluateIndeterminates(x, grid_x_vals).reshape(grid_theta.shape)
    grid_b = b.EvaluateIndeterminates(x, grid_x_vals).reshape(grid_theta.shape)

    # We draw the region that grid_incompatible_vals > 0 and grid_V < rho and grid_b > 0
    grid_error_value = np.min(
        np.concatenate(
            (
                np.expand_dims(grid_incompatible_vals, axis=2),
                np.expand_dims(rho - grid_V, axis=2),
                np.expand_dims(grid_b, axis=2),
            ),
            axis=2,
        ),
        axis=2,
    )
    h_error = ax.contourf(
        grid_theta,
        grid_x2,
        grid_error_value,
        levels=[0, np.inf],
        colors="purple",
        alpha=0.2,
    )
    return h_error


def visualize():
    kappa_V = 0.1
    x = sym.MakeVectorContinuousVariable(3, "x")
    V = search_clf(x, kappa_V)
    b = get_cbf_init(x)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel(r"$\theta$ (rad)", fontsize=16)
    ax.set_ylabel(r"$\gamma$", fontsize=16)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"0", r"$\frac{\pi}{2}$", r"$\pi$"]
    )
    ax.tick_params(axis="both", which="major", labelsize=14)
    h_V, h_b, h_compatible = demo_trigpoly.plot_clf_cbf(
        ax, V, np.array([b]), x, fill_compatible=False
    )
    demo_trigpoly.plot_unsafe_regions(ax)
    h_incompatible = demo_trigpoly.plot_incompatible(  # noqa
        ax,
        V,
        b,
        x,
        kappa_V=kappa_V,
        kappa_b=0.1,
    )
    ax.legend(
        [
            h_V.legend_elements()[0][0],
            h_b.legend_elements()[0][0],
        ],
        [r"$V(x)=1$", r"$h(x)=0$"],
        prop={"size": 12},
    )
    ax.set_title("Incompatible CLF/CBF")
    fig.show()
    for fig_extension in (".png", ".pdf"):
        fig.savefig(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"../../figures/nonlinear_toy_incompatible{fig_extension}",
            ),
            bbox_inches="tight",
        )

    pass


def main():
    visualize()


if __name__ == "__main__":
    main()
