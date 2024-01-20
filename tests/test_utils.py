import compatible_clf_cbf.utils as mut

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


def test_to_lower_triangular_columns():
    vec = mut.to_lower_triangular_columns(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    np.testing.assert_equal(vec, np.array([1, 4, 7, 5, 8, 9]))


class TestContainmentLagrangian:
    @pytest.mark.parametrize(
        "inner_ineq_degree,outer_degree", [([0], 0), ([0], -1), ([-1], 0)]
    )
    def test_add_constraint1(self, inner_ineq_degree, outer_degree):
        x = sym.MakeVectorContinuousVariable(2, "x")
        x_set = sym.Variables(x)
        inner_ineq_poly = np.array([sym.Polynomial(x[0] ** 2 + x[1] ** 2 - 1)])
        outer_poly = sym.Polynomial(x[0] ** 2 + x[1] ** 2 - 1.01)
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(x_set)
        containment_lagrangian_degree = mut.ContainmentLagrangianDegree(
            inner_ineq=inner_ineq_degree, inner_eq=[], outer=outer_degree
        )
        lagrangians = containment_lagrangian_degree.construct_lagrangian(prog, x_set)
        lagrangians.add_constraint(
            prog, inner_ineq_poly, inner_eq_poly=np.array([]), outer_poly=outer_poly
        )
        result = solvers.Solve(prog)
        assert result.is_success()

    @pytest.mark.parametrize(
        "inner_ineq_degree,outer_degree", [([0], 0), ([0], -1), ([-1], 0)]
    )
    def test_add_constraint2(self, inner_ineq_degree, outer_degree):
        # The inner_poly sub-level set is larger than the outer_poly sub-level
        # set. The program should fail.
        x = sym.MakeVectorContinuousVariable(2, "x")
        x_set = sym.Variables(x)
        inner_ineq_poly = np.array([sym.Polynomial(x[0] ** 2 + x[1] ** 2 - 1.01)])
        outer_poly = sym.Polynomial(x[0] ** 2 + x[1] ** 2 - 1.0)
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(x_set)
        containment_lagrangian_degree = mut.ContainmentLagrangianDegree(
            inner_ineq=inner_ineq_degree, inner_eq=[], outer=outer_degree
        )
        lagrangians = containment_lagrangian_degree.construct_lagrangian(prog, x_set)
        lagrangians.add_constraint(
            prog, inner_ineq_poly, inner_eq_poly=None, outer_poly=outer_poly
        )
        result = solvers.Solve(prog)
        assert not result.is_success()

    def test_add_constraint3(self):
        """
        The inner algebraic set has multiple polynomials, with both equalities
        and inequalities.
        """
        x = sym.MakeVectorContinuousVariable(2, "x")
        x_set = sym.Variables(x)
        inner_ineq_poly = np.array(
            [
                sym.Polynomial(x[0] ** 2 + x[1] ** 2 - 1),
                sym.Polynomial(0.5 - x[0] ** 2 - 2 * x[1] ** 2),
            ]
        )
        inner_eq_poly = np.array([sym.Polynomial(x[0] + x[1])])
        outer_poly = sym.Polynomial(x[0] ** 2 + 0.5 * x[1] ** 2 - 1)

        lagrangian_degrees = mut.ContainmentLagrangianDegree(
            inner_ineq=[0, 0], inner_eq=[1], outer=0
        )
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(x_set)
        lagrangians = lagrangian_degrees.construct_lagrangian(prog, x_set)
        lagrangians.add_constraint(prog, inner_ineq_poly, inner_eq_poly, outer_poly)
        result = solvers.Solve(prog)
        assert result.is_success()
        lagrangians_result = lagrangians.get_result(result)
        assert lagrangians_result.inner_ineq[0].TotalDegree() == 0
        assert lagrangians_result.inner_ineq[1].TotalDegree() == 0
        assert lagrangians_result.inner_eq[0].TotalDegree() == 1
        assert lagrangians_result.outer.TotalDegree() == 0
