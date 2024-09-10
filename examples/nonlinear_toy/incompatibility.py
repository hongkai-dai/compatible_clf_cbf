import numpy as np
import pydrake.symbolic as sym
import pydrake.solvers as solvers

import compatible_clf_cbf.clf as clf
import compatible_clf_cbf.cbf as cbf
import examples.nonlinear_toy.toy_system as toy_system


class ComputeIncompatibility:
    """
    The CLF constraint is a_V * u <= b_V
    The CBF constraint is a_h * u <= b_h
    We define the incompatibility as the "signed distance" between the the set
    {u | a_V * u <= b_V} and the set {u | a_h * u <= b_h}.
    """

    def __init__(
        self,
        V: sym.Polynomial,
        h: sym.Polynomial,
        x: np.ndarray,
        kappa_V: float,
        kappa_h: float,
        u_min: float,
        u_max: float,
    ):
        f, g = toy_system.affine_trig_poly_dynamics(x)
        self.clf_constraint = clf.ClfConstraint(V, f, g, x, kappa_V)
        self.cbf_constraint = cbf.CbfConstraint(h, f, g, x, kappa_h)
        self.u_min = u_min
        self.u_max = u_max

    def eval(self, x_val: np.ndarray) -> float:
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(1, "u")
        clf_cnstr = self.clf_constraint.add_to_prog(prog, x_val, u)
        cbf_cnstr = self.cbf_constraint.add_to_prog(prog, x_val, u)
        # Write clf_cnstr in the a_V * u <= b_V form
        if np.isinf(clf_cnstr.evaluator().upper_bound()):
            a_V = -clf_cnstr.evaluator().GetDenseA()[0, 0]
            b_V = -clf_cnstr.evaluator().lower_bound()[0]
        else:
            a_V = clf_cnstr.evaluator().GetDenseA()[0, 0]
            b_V = clf_cnstr.evaluator().upper_bound()[0]
        # Write cbf_cnstr in the a_h * u <= b_h form
        if np.isinf(cbf_cnstr.evaluator().upper_bound()):
            a_h = -cbf_cnstr.evaluator().GetDenseA()[0, 0]
            b_h = -cbf_cnstr.evaluator().lower_bound()[0]
        else:
            a_h = cbf_cnstr.evaluator().GetDenseA()[0, 0]
            b_h = cbf_cnstr.evaluator().upper_bound()[0]
        if a_V > 0:
            # By CLF constraint, we have u <= b_V / a_V
            u_clf_upper = b_V / a_V
            u_clf_lower = -np.inf
        elif a_V == 0:
            if b_V >= 0:
                u_clf_upper = np.inf
                u_clf_lower = -np.inf
            else:
                u_clf_upper = -np.inf
                u_clf_lower = -np.inf
        else:
            u_clf_upper = np.inf
            u_clf_lower = b_V / a_V

        if a_h > 0:
            u_cbf_upper = b_h / a_h
            u_cbf_lower = -np.inf
        elif a_h == 0:
            if b_h >= 0:
                u_cbf_upper = np.inf
                u_cbf_lower = -np.inf
            else:
                u_cbf_upper = -np.inf
                u_cbf_lower = -np.inf
        else:
            u_cbf_upper = np.inf
            u_cbf_lower = b_h / a_h

        u_upper = np.min([u_cbf_upper, u_clf_upper, self.u_max])
        u_lower = np.max([u_clf_lower, u_cbf_lower, self.u_min])

        if not np.isinf(u_upper) and not np.isinf(u_lower):
            incompatible = u_lower - u_upper
        elif np.isinf(u_upper) and not np.isinf(u_lower):
            incompatible = -np.abs(u_clf_lower - u_cbf_lower)
        elif np.isinf(u_lower) and not np.isinf(u_upper):
            incompatible = -np.abs(u_clf_upper - u_clf_upper)
        else:
            incompatible = -np.inf
        return incompatible
