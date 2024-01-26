"""
Find the CLF and CBF jointly, versus finding CLF and CBF separately for a
2-dimensional single-integrator system xdot = u. I assume the obstacle is a
sphere.
"""
from typing import Tuple

import numpy as np

import pydrake.symbolic as sym

from compatible_clf_cbf.clf import ControlLyapunov, ClfWoInputLimitLagrangianDegrees
from compatible_clf_cbf.cbf import ControlBarrier, CbfWoInputLimitLagrangianDegrees
import compatible_clf_cbf.clf_cbf as clf_cbf


def affine_dynamics() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the dynamics as f(x) + g(x)*u
    """
    f = np.array([sym.Polynomial(0), sym.Polynomial(0)])
    g = np.array(
        [[sym.Polynomial(1), sym.Polynomial(0)], [sym.Polynomial(0), sym.Polynomial(1)]]
    )
    return f, g


def get_unsafe_region(
    x: np.ndarray, obstacle_center: np.ndarray, obstacle_radius: float
) -> np.ndarray:
    return np.array(
        [
            sym.Polynomial(
                (x[0] - obstacle_center[0]) ** 2
                + (x[1] - obstacle_center[1]) ** 2
                - obstacle_radius**2
            )
        ]
    )


def find_clf_cbf_separately(
    x: np.ndarray, obstacle_center: np.ndarray, obstacle_radius: float
) -> Tuple[sym.Polynomial, sym.Polynomial]:
    assert x.shape == (2,)
    clf = sym.Polynomial(x[0] ** 2 + x[1] ** 2)
    assert obstacle_center.shape == (2,)
    cbf = sym.Polynomial(
        (x[0] - obstacle_center[0]) ** 2
        + (x[1] - obstacle_center[1]) ** 2
        - obstacle_radius**2
    )
    return clf, cbf


def certify_clf_cbf_separately(
    x: np.ndarray,
    clf: sym.Polynomial,
    cbf: sym.Polynomial,
    rho: float,
    kappa_V: float,
    kappa_b: float,
    barrier_eps: float,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
):
    f, g = affine_dynamics()
    control_lyapunov = ControlLyapunov(
        f=f,
        g=g,
        x=x,
        x_equilibrium=np.zeros(2),
        u_vertices=None,
        state_eq_constraints=None,
    )
    clf_lagrangian = control_lyapunov.search_lagrangian_given_clf(
        clf,
        rho,
        kappa_V,
        ClfWoInputLimitLagrangianDegrees(
            dVdx_times_f=0,
            dVdx_times_g=[1, 1],
            rho_minus_V=0,
            state_eq_constraints=None,
        ),
    )
    assert clf_lagrangian is not None

    control_barrier = ControlBarrier(
        f=f,
        g=g,
        x=x,
        unsafe_region=get_unsafe_region(x, obstacle_center, obstacle_radius),
        u_vertices=None,
        state_eq_constraints=None,
    )
    (
        cbf_derivative_lagrangians,
        unsafe_lagrangians,
    ) = control_barrier.search_lagrangians_given_cbf(
        cbf,
        barrier_eps,
        kappa_b,
        CbfWoInputLimitLagrangianDegrees(
            dbdx_times_f=0, dbdx_times_g=[1, 1], b_plus_eps=0, state_eq_constraints=None
        ),
        clf_cbf.UnsafeRegionLagrangianDegrees(
            cbf=0, unsafe_region=[0], state_eq_constraints=None
        ),
    )
    assert cbf_derivative_lagrangians is not None
    assert unsafe_lagrangians is not None

    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        unsafe_regions=[get_unsafe_region(x, obstacle_center, obstacle_radius)],
        Au=None,
        bu=None,
        with_clf=True,
        use_y_squared=True,
        state_eq_constraints=None,
    )
    x_incompatible = (
        obstacle_center
        + obstacle_center / np.linalg.norm(obstacle_center) * obstacle_radius
    )
    assert not compatible.check_compatible_at_state(
        clf, np.array([cbf]), x_incompatible, kappa_V, np.array([kappa_b])
    )[0]


def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    obstacle_center = np.array([1.0, 0.0])
    obstacle_radius = 0.5
    V, b = find_clf_cbf_separately(x, obstacle_center, obstacle_radius)
    rho = 10
    kappa_V = 0.01
    kappa_b = 0.01
    barrier_eps = 0.0
    certify_clf_cbf_separately(
        x, V, b, rho, kappa_V, kappa_b, barrier_eps, obstacle_center, obstacle_radius
    )


if __name__ == "__main__":
    main()
