import compatible_clf_cbf.utils as mut

from typing import Optional

import numpy as np
import pytest  # noqa

import pydrake.symbolic as sym
import pydrake.solvers as solvers


def test_check_array_of_polynomials():
    x = sym.MakeVectorContinuousVariable(rows=3, name="x")
    x_set = sym.Variables(x)
    p = np.array([sym.Polynomial(x[0] * x[0]), sym.Polynomial(x[1] + 2)])
    mut.check_array_of_polynomials(p, x_set)


def check_psd(X: np.ndarray, tol: float):
    assert np.all((np.linalg.eig(X)[0] >= -tol))


def test_add_log_det_lower():
    def tester(lower):
        prog = solvers.MathematicalProgram()
        X = prog.NewSymmetricContinuousVariables(3, "X")
        ret = mut.add_log_det_lower(prog, X, lower)
        result = solvers.Solve(prog)
        assert result.is_success()
        X_sol = result.GetSolution(X)
        check_psd(X_sol, tol=1e-6)
        assert np.log(np.linalg.det(X_sol)) >= lower - 1e-6
        t_sol = result.GetSolution(ret.t)
        assert np.sum(t_sol) >= lower - 1e-6

    tester(2.0)
    tester(3.0)


def is_sos(
    poly: sym.Polynomial,
    solver_id: Optional[solvers.SolverId] = None,
    solver_options: Optional[solvers.SolverOptions] = None,
):
    prog = solvers.MathematicalProgram()
    prog.AddIndeterminates(poly.indeterminates())
    assert poly.decision_variables().empty()
    prog.AddSosConstraint(poly)
    if solver_id is None:
        result = solvers.Solve(prog, None, solver_options)
    else:
        solver = solvers.MakeSolver(solver_id)
        result = solver.Solve(prog, None, solver_options)
    return result.is_success()
