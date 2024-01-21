"""
Find the CLF/CBF without the input limits.
We use the trigonometric state with polynomial dynamics.
"""

import numpy as np

import pydrake.solvers as solvers
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
    V_init = sym.Polynomial(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    b_init = np.array([sym.Polynomial(0.1 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2)])

    lagrangian_degrees = clf_cbf.CompatibleLagrangianDegrees(
        lambda_y=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0)],
        xi_y=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=0),
        y=None,
        rho_minus_V=clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2),
        b_plus_eps=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2)],
        state_eq_constraints=[clf_cbf.CompatibleLagrangianDegrees.Degree(x=2, y=2)],
    )
    rho = 0.1
    kappa_V = 0.01
    kappa_b = np.array([0.01])
    barrier_eps = np.array([0.001])
    (
        compatible_prog,
        compatible_lagrangians,
    ) = compatible.construct_search_compatible_lagrangians(
        V_init,
        b_init,
        kappa_V,
        kappa_b,
        lagrangian_degrees,
        rho,
        barrier_eps,
    )
    assert compatible_lagrangians is not None
    solver_options = solvers.SolverOptions()
    solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 0)
    compatible_result = solvers.Solve(compatible_prog, None, solver_options)
    assert compatible_result.is_success()


if __name__ == "__main__":
    main()
