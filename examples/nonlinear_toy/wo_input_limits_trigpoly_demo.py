"""
Find the CLF/CBF without the input limits.
We use the trigonometric state with polynomial dynamics.
"""

import numpy as np

import pydrake.symbolic as sym

import compatible_clf_cbf.clf_cbf as clf_cbf
import examples.nonlinear_toy.toy_system as toy_system


def main():
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = toy_system.affine_trig_poly_dynamics(x)
    state_eq_constraints = np.array([toy_system.affine_trig_poly_state_constraints(x)])
    use_y_squared = True
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        unsafe_regions=[np.array([sym.Polynomial(x[0] + x[1] + x[2] + 3)])],
        Au=None,
        bu=None,
        with_clf=True,
        use_y_squared=use_y_squared,
        state_eq_constraints=state_eq_constraints,
    )
    V_init = sym.Polynomial(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) / 0.1
    b_init = np.array([sym.Polynomial(0.1 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2)])

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
    max_iter = 5

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
        x_inner=x_equilibrium,
        find_inner_ellipsoid_max_iter=1,
    )
    print(f"V={V}")
    print(f"b={b}")


if __name__ == "__main__":
    main()
