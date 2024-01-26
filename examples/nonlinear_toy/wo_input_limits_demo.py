"""
Find the compatible CLF/CBF with no input limits.
"""
import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_cbf import clf_cbf
from examples.nonlinear_toy import toy_system


def main(use_y_squared: bool):
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = toy_system.affine_dynamics(x)
    use_y_squared = True
    compatible = clf_cbf.CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        unsafe_regions=[np.array([sym.Polynomial(x[0] + 10)])],
        Au=None,
        bu=None,
        with_clf=True,
        use_y_squared=use_y_squared,
    )
    V_init = sym.Polynomial(x[0] ** 2 + x[1] ** 2)
    b_init = np.array([sym.Polynomial(0.001 - x[0] ** 2 - x[1] ** 2)])
    kappa_V = 1e-3
    kappa_b = np.array([kappa_V])

    lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=3, y=0)],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0),
        y=None
        if use_y_squared
        else [
            clf_cbf.CompatibleLagrangianDegrees.Degree(x=4, y=0)
            for _ in range(compatible.y.size)
        ],
        rho_minus_V=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0),
        b_plus_eps=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0)],
        state_eq_constraints=None,
    )
    rho = 0.01
    barrier_eps = np.array([0.0001])
    (
        compatible_prog,
        compatible_lagrangians,
    ) = compatible.construct_search_compatible_lagrangians(
        V_init, b_init, kappa_V, kappa_b, lagrangian_degrees, rho, barrier_eps
    )
    compatible_result = solvers.Solve(compatible_prog)
    assert compatible_result.is_success()

    unsafe_region_lagrangian_degrees = clf_cbf.UnsafeRegionLagrangianDegrees(
        cbf=0, unsafe_region=[0], state_eq_constraints=None
    )

    unsafe_region_lagrangians = compatible.certify_cbf_unsafe_region(
        unsafe_region_index=0,
        cbf=b_init[0],
        lagrangian_degrees=unsafe_region_lagrangian_degrees,
        solver_options=None,
    )
    assert unsafe_region_lagrangians is not None


if __name__ == "__main__":
    main(use_y_squared=True)
    main(use_y_squared=False)
