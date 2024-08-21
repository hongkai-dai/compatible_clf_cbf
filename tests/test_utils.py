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


def test_solve_w_id():
    prog = solvers.MathematicalProgram()
    x = prog.NewContinuousVariables(2)
    prog.AddBoundingBoxConstraint(-1, 1, x)
    prog.AddLinearCost(x[0] + x[1] + 1)
    result = mut.solve_with_id(
        prog, solver_id=None, solver_options=None, backoff_rel_scale=0.1
    )
    assert result.is_success()
    # I know the optimal solution is obtained at (-1, -1), with the optimal cost being
    # -1. Hence by backing off, the solution should satisfy x[0] + x[1] + 1 <= -0.9
    x_sol = result.GetSolution(x)
    assert x_sol[0] + x_sol[1] + 1 <= -0.9 + 1e-5
    # Now add the objective max x[0] + x[1]. The maximazation should be
    # x[0] + x[1] = -1.9
    prog.AddLinearCost(-x[0] - x[1])
    result = mut.solve_with_id(
        prog, solver_id=None, solver_options=None, backoff_rel_scale=None
    )
    x_sol = result.GetSolution(x)
    np.testing.assert_allclose(x_sol[0] + x_sol[1], -1.9, atol=1e-5)

    # Now test the problem with a positive optimal cost.
    prog = solvers.MathematicalProgram()
    x = prog.NewContinuousVariables(2)
    prog.AddBoundingBoxConstraint(-1, 1, x)
    prog.AddLinearCost(x[0] + x[1] + 3)
    result = mut.solve_with_id(
        prog, solver_id=None, solver_options=None, backoff_rel_scale=0.1
    )
    assert result.is_success()
    # I know the optimal solution is obtained at (-1, -1), with the optimal cost being
    # 1. Hence by backing off, the solutionshould satisfy x[0] + x[1] + 3 <= 1.1
    x_sol = result.GetSolution(x)
    assert x_sol[0] + x_sol[1] + 3 <= 1.1 + 1e-5
    # Now add the objective max x[0] + x[1]. The maximization should be
    # x[0] + x[1] = -1.9
    prog.AddLinearCost(-x[0] - x[1])
    result = mut.solve_with_id(
        prog, solver_id=None, solver_options=None, backoff_rel_scale=None
    )
    x_sol = result.GetSolution(x)
    np.testing.assert_allclose(x_sol[0] + x_sol[1], -1.9, atol=1e-5)


def test_find_no_linear_term_variables():
    x = sym.MakeVectorContinuousVariable(4, "x")
    x_set = sym.Variables(x)
    p = np.array(
        [
            sym.Polynomial(x[0] ** 2 + x[1] ** 2 + x[1] * x[3] + x[2] + 1),
            sym.Polynomial(x[0] ** 3 - 2 * x[3]),
        ]
    )
    no_linear_term_variables = mut.find_no_linear_term_variables(x_set, p)
    assert no_linear_term_variables.size() == 2
    assert no_linear_term_variables.include(x[0])
    assert no_linear_term_variables.include(x[1])
    assert x_set.size() == 4


def test_new_free_polynomial_pass_origin():
    prog = solvers.MathematicalProgram()
    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)
    degree = 2
    coeff_name = "a"
    no_linear_term_variables = sym.Variables(np.array([x[1], x[2]]))
    p = mut.new_free_polynomial_pass_origin(
        prog, x_set, degree, coeff_name, no_linear_term_variables
    )
    assert sym.Monomial(x[0], 2) in p.monomial_to_coefficient_map().keys()
    assert sym.Monomial(x[1], 2) in p.monomial_to_coefficient_map().keys()
    assert sym.Monomial(x[2], 2) in p.monomial_to_coefficient_map().keys()
    assert sym.Monomial({x[0]: 1, x[1]: 1}) in p.monomial_to_coefficient_map().keys()
    assert sym.Monomial({x[0]: 1, x[2]: 1}) in p.monomial_to_coefficient_map().keys()
    assert sym.Monomial({x[1]: 1, x[2]: 1}) in p.monomial_to_coefficient_map().keys()
    assert sym.Monomial(x[0]) in p.monomial_to_coefficient_map().keys()
    assert len(p.monomial_to_coefficient_map().keys()) == 7


def test_serialize_polynomial():
    def test(p, x_set, expected):
        ret = mut.serialize_polynomial(p, x_set)
        assert len(ret) == len(expected)
        for m, c in ret.items():
            assert m in expected.keys()
            assert c == expected[m]

    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)
    test(sym.Polynomial(3 * x[0] ** 2), None, {(2,): 3})
    test(sym.Polynomial(3 * x[0] ** 2), x_set, {(2, 0, 0): 3})
    test(
        sym.Polynomial(3 * x[0] ** 2 + 2 * x[1] * x[2]),
        x_set,
        {(2, 0, 0): 3, (0, 1, 1): 2},
    )


def test_deserialize_polynomial():
    def test(monomial_degrees_to_coefficient, x, p_expected):
        p = mut.deserialize_polynomial(monomial_degrees_to_coefficient, x)
        assert p.EqualTo(p_expected)
        p_again = mut.deserialize_polynomial(mut.serialize_polynomial(p, x), x)
        assert p_again.EqualTo(p_expected)

    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)
    test({(2, 0, 1): 4}, x_set, sym.Polynomial(4 * x[0] ** 2 * x[2]))
    test(
        {(2, 0, 1): 4, (0, 2, 3): -1},
        x_set,
        sym.Polynomial(4 * x[0] ** 2 * x[2] - x[1] ** 2 * x[2] ** 3),
    )
