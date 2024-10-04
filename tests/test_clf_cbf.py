import compatible_clf_cbf.clf_cbf as mut

from typing import List, Optional, Tuple

import numpy as np
import pytest  # noqa

import pydrake.solvers as solvers
import pydrake.symbolic as sym

import compatible_clf_cbf.ellipsoid_utils as ellipsoid_utils
import compatible_clf_cbf.utils as utils


class TestCompatibleLagrangianDegrees(object):
    def test_construct_polynomial(self):
        degree = mut.XYDegree(x=3, y=2)
        prog = solvers.MathematicalProgram()
        x = prog.NewIndeterminates(3, "x")
        y = prog.NewIndeterminates(2, "y")

        sos_poly = degree.construct_polynomial(
            prog, sym.Variables(x), sym.Variables(y), is_sos=True
        )
        assert len(prog.positive_semidefinite_constraints()) == 1
        free_poly = degree.construct_polynomial(
            prog, sym.Variables(x), sym.Variables(y), is_sos=False
        )
        assert len(prog.positive_semidefinite_constraints()) == 1
        for monomial, _ in sos_poly.monomial_to_coefficient_map().items():
            # The total degree for x should be <= 2 (the largest even degree <= 3).
            assert np.sum([monomial.degree(x[i]) for i in range(x.size)]) <= 2
            assert np.sum([monomial.degree(y[i]) for i in range(y.size)]) <= 2
        for monomial, _ in free_poly.monomial_to_coefficient_map().items():
            # The total degree for x should be <= 3
            assert np.sum([monomial.degree(x[i]) for i in range(x.size)]) <= 3
            assert np.sum([monomial.degree(y[i]) for i in range(y.size)]) <= 2


class TestCompatibleStatesOptions:
    def test_add_cost(self):
        dut = mut.CompatibleStatesOptions(
            candidate_compatible_states=np.array([[0.2, 0.5], [-0.1, 1.2], [0.3, 2]]),
            anchor_states=None,
            h_anchor_bounds=None,
            weight_V=1.5,
            weight_h=np.array([1.2, 1.5]),
            V_margin=0.1,
            h_margins=np.array([0.2, 0.3]),
        )
        prog = solvers.MathematicalProgram()
        x = prog.NewIndeterminates(2, "x")
        x_set = sym.Variables(x)
        V = prog.NewFreePolynomial(x_set, 2)
        h = np.array([prog.NewFreePolynomial(x_set, 3) for i in range(2)])
        cost, V_relu, h_relu = dut.add_cost(prog, x, V, h)
        assert V_relu is not None

        def check_feasible(
            V_val: sym.Polynomial,
            h_val: np.ndarray,
        ):
            constraint1 = prog.AddEqualityConstraintBetweenPolynomials(V, V_val)
            constraint2 = prog.AddEqualityConstraintBetweenPolynomials(h[0], h_val[0])
            constraint3 = prog.AddEqualityConstraintBetweenPolynomials(h[1], h_val[1])
            result = solvers.Solve(prog)

            V_relu_expected = np.maximum(
                V_val.EvaluateIndeterminates(x, dut.candidate_compatible_states.T)
                - (1 - (0 if dut.V_margin is None else dut.V_margin)),
                np.zeros(dut.candidate_compatible_states.shape[0]),
            )
            h_relu_expected = np.array(
                [
                    np.maximum(
                        -h_val[i].EvaluateIndeterminates(
                            x, dut.candidate_compatible_states.T
                        )
                        + (0 if dut.h_margins is None else dut.h_margins[i]),
                        np.zeros(dut.candidate_compatible_states.shape[0]),
                    ).reshape((-1,))
                    for i in range(2)
                ]
            )
            np.testing.assert_allclose(result.GetSolution(V_relu), V_relu_expected)
            np.testing.assert_allclose(result.GetSolution(h_relu), h_relu_expected)

            cost_expected = dut.weight_V * V_relu_expected.sum() + dut.weight_h.dot(
                h_relu_expected.sum(axis=1)
            )
            np.testing.assert_allclose(result.get_optimal_cost(), cost_expected)
            for constraints in [constraint1, constraint2, constraint3]:
                for c in constraints:
                    prog.RemoveConstraint(c)

        check_feasible(
            sym.Polynomial(2 * x[0] * x[0] + 3 * x[1] * x[0] + 2 + 2 * x[1]),
            np.array(
                [
                    sym.Polynomial(x[0] * x[1] * x[0] + 3),
                    sym.Polynomial(x[0] * x[1] * x[1] - x[0] ** 3 + 2 * x[0] + 1),
                ]
            ),
        )
        check_feasible(
            sym.Polynomial(-2 * x[0] * x[0] + 3 * x[1] * x[0] + 2 + 2 * x[1]),
            np.array(
                [
                    sym.Polynomial(x[0] * x[1] + 3),
                    sym.Polynomial(x[0] * x[1] * x[1] - x[0] ** 3 + 2 * x[0] + 1),
                ]
            ),
        )

    def test_add_constraint(self):
        dut = mut.CompatibleStatesOptions(
            candidate_compatible_states=np.array([[0.2, 0.5], [-0.1, 1.2], [0.3, 2]]),
            anchor_states=np.array([[0.5, 0.3], [0.2, 0.1], [0.9, 2]]),
            h_anchor_bounds=[(np.array([-0.5, 0.3, -3]), np.array([1, 4, 0.5]))],
            weight_V=1.5,
            weight_h=np.array([1.2, 1.5]),
        )

        prog = solvers.MathematicalProgram()
        x = prog.NewIndeterminates(2, "x")
        x_set = sym.Variables(x)
        h = np.array([prog.NewFreePolynomial(x_set, 2)])

        constraints = dut.add_constraint(prog, x, h)
        assert constraints is not None
        assert len(constraints) == 1

        result = solvers.Solve(prog)
        assert result.is_success()
        h_result = np.array([result.GetSolution(h_i) for h_i in h])
        assert dut.anchor_states is not None
        h_result_at_anchor = h_result[0].EvaluateIndeterminates(x, dut.anchor_states.T)
        assert dut.h_anchor_bounds is not None
        assert np.all(h_result_at_anchor >= dut.h_anchor_bounds[0][0] - 1e-6)
        assert np.all(h_result_at_anchor <= dut.h_anchor_bounds[0][1] + 1e-6)


class TestClfCbf(object):
    @classmethod
    def setup_class(cls):
        cls.nx = 3
        cls.nu = 2
        cls.x = sym.MakeVectorContinuousVariable(cls.nx, "x")
        # The dynamics and safety sets are arbitrary, they are only used to
        # test the functionality of the code.
        cls.f = np.array(
            [
                sym.Polynomial(cls.x[0] ** 3),
                sym.Polynomial(-cls.x[1] * cls.x[0]),
                sym.Polynomial(cls.x[0] + cls.x[2]),
            ]
        )
        cls.g = np.array(
            [
                [sym.Polynomial(cls.x[0] ** 2 + cls.x[1] ** 2), sym.Polynomial()],
                [sym.Polynomial(), sym.Polynomial(cls.x[1] * cls.x[2])],
                [sym.Polynomial(cls.x[0] * cls.x[2]), sym.Polynomial(cls.x[1])],
            ]
        )
        cls.exclude_sets = [
            mut.ExcludeSet(np.array([sym.Polynomial(cls.x[0] + 1)])),
            mut.ExcludeSet(
                np.array([sym.Polynomial(1 - cls.x[1]), sym.Polynomial(1 - cls.x[0])])
            ),
        ]
        cls.within_set = None

    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the dynamics at x=0 and u = 0
        """
        env = {self.x[i]: 0 for i in range(self.nx)}
        for i in range(self.nx):
            assert self.f[i].Evaluate(env) == 0
        A = np.empty((self.nx, self.nx))
        B = np.empty((self.nx, self.nu))
        for i in range(self.nx):
            for j in range(self.nx):
                A[i, j] = self.f[i].Differentiate(self.x[j]).Evaluate(env)
            for j in range(self.nu):
                B[i, j] = self.g[i, j].Evaluate(env)
        return (A, B)

    def test_constructor(self):
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=1,
            with_clf=True,
            use_y_squared=True,
        )

        def check_members(cls: mut.CompatibleClfCbf):
            assert cls.y_poly.shape == cls.y.shape
            for y_poly_i, y_i in zip(cls.y_poly.flat, cls.y.flat):
                assert y_poly_i.EqualTo(sym.Polynomial(y_i))
            assert cls.y_squared_poly.shape == cls.y.shape
            for y_squared_poly_i, y_i in zip(cls.y_squared_poly.flat, cls.y.flat):
                assert y_squared_poly_i.EqualTo(sym.Polynomial(y_i**2))
            assert cls.y_cross_poly.size == cls.y.size * (cls.y.size - 1) / 2
            for y_cross in cls.y_cross_poly:
                assert y_cross.TotalDegree() == 2
                assert len(y_cross.monomial_to_coefficient_map()) == 1
                for v in y_cross.indeterminates():
                    assert y_cross.Degree(v) == 1

        assert dut.y.shape == (dut.num_cbf + 1,)
        check_members(dut)

        # Now construct with with_clf=False
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=1,
            with_clf=False,
            use_y_squared=True,
        )
        assert dut.y.shape == (dut.num_cbf,)
        check_members(dut)

        # Now construct with Au and bu
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=np.array([[-3, -2], [1.0, 4.0], [0.0, 3.0]]),
            bu=np.array([4, 5.0, 6.0]),
            num_cbf=2,
            with_clf=False,
            use_y_squared=True,
        )
        assert dut.Au is not None
        assert dut.Au.shape == (3, self.nu)
        assert dut.bu is not None
        assert dut.bu.shape == (3,)
        assert dut.y.shape == (dut.num_cbf + dut.Au.shape[0],)
        check_members(dut)

        # Now construct with Au, bu and with_clf=True
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=np.array([[-3, -2], [1, 4], [3, -1.0]]),
            bu=np.array([4, 5.0, 6.0]),
            num_cbf=1,
            with_clf=True,
            use_y_squared=True,
        )
        assert dut.Au is not None
        assert dut.Au.shape == (3, self.nu)
        assert dut.bu is not None
        assert dut.bu.shape == (3,)
        assert dut.y.shape == (dut.num_cbf + 1 + dut.Au.shape[0],)
        check_members(dut)

    def test_calc_xi_Lambda_w_clf(self):
        """
        Test _calc_xi_Lambda with CLF.
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=2,
            with_clf=True,
            use_y_squared=True,
        )
        V = sym.Polynomial(
            self.x[0] ** 2 + self.x[1] ** 2 + self.x[2] ** 2 + self.x[0] * 2
        )
        h = np.array(
            [
                sym.Polynomial(1 - self.x[0] ** 2 - self.x[1] ** 2 - self.x[2] ** 2),
                sym.Polynomial(2 - self.x[0] ** 4 - self.x[2] ** 2 * self.x[1] ** 2),
            ]
        )
        kappa_V = 0.01
        kappa_h = np.array([0.02, 0.03])
        xi, lambda_mat = dut._calc_xi_Lambda(V=V, h=h, kappa_V=kappa_V, kappa_h=kappa_h)
        assert xi.shape == (1 + dut.num_cbf,)
        assert lambda_mat.shape == (1 + dut.num_cbf, dut.nu)
        dhdx = np.empty((2, 3), dtype=object)
        dhdx[0] = h[0].Jacobian(self.x)
        dhdx[1] = h[1].Jacobian(self.x)
        dVdx = V.Jacobian(self.x)
        xi_expected = np.empty((3,), dtype=object)
        xi_expected[0] = dhdx[0] @ self.f + kappa_h[0] * h[0]
        xi_expected[1] = dhdx[1] @ self.f + kappa_h[1] * h[1]
        xi_expected[2] = -dVdx @ self.f - kappa_V * V
        utils.check_polynomial_arrays_equal(xi, xi_expected, 1e-8)

        lambda_mat_expected = np.empty((3, self.nu), dtype=object)
        lambda_mat_expected[:2] = -dhdx @ self.g
        lambda_mat_expected[-1] = dVdx @ self.g
        utils.check_polynomial_arrays_equal(lambda_mat, lambda_mat_expected, 1e-8)

    def test_calc_xi_Lambda_wo_clf(self):
        """
        Test _calc_xi_Lambda without CLF.
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=2,
            with_clf=False,
            use_y_squared=True,
        )
        V = None
        h = np.array(
            [
                sym.Polynomial(1 - self.x[0] ** 2 - self.x[1] ** 2 - self.x[2] ** 2),
                sym.Polynomial(2 - self.x[0] ** 4 - self.x[2] ** 2 * self.x[1] ** 2),
            ]
        )
        kappa_V = None
        kappa_h = np.array([0.02, 0.03])
        xi, lambda_mat = dut._calc_xi_Lambda(V=V, h=h, kappa_V=kappa_V, kappa_h=kappa_h)
        assert xi.shape == (dut.num_cbf,)
        assert lambda_mat.shape == (dut.num_cbf, dut.nu)
        dhdx = np.empty((2, 3), dtype=object)
        dhdx[0] = h[0].Jacobian(self.x)
        dhdx[1] = h[1].Jacobian(self.x)
        xi_expected = np.empty((2,), dtype=object)
        xi_expected[0] = dhdx[0] @ self.f + kappa_h[0] * h[0]
        xi_expected[1] = dhdx[1] @ self.f + kappa_h[1] * h[1]
        utils.check_polynomial_arrays_equal(xi, xi_expected, 1e-8)

        lambda_mat_expected = np.empty((2, self.nu), dtype=object)
        lambda_mat_expected = -dhdx @ self.g
        utils.check_polynomial_arrays_equal(lambda_mat, lambda_mat_expected, 1e-8)

    def test_calc_xi_Lambda_w_clf_Aubu(self):
        """
        Test _calc_xi_Lambda with CLF and Au * u <= bu
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=np.array([[-3, 2], [1, 4], [3, 8.0]]),
            bu=np.array([3.0, 5.0, 10.0]),
            num_cbf=2,
            with_clf=True,
            use_y_squared=True,
        )
        V = sym.Polynomial(
            self.x[0] ** 2 + self.x[1] ** 2 + self.x[2] ** 2 + self.x[0] * 2
        )
        h = np.array(
            [
                sym.Polynomial(1 - self.x[0] ** 2 - self.x[1] ** 2 - self.x[2] ** 2),
                sym.Polynomial(2 - self.x[0] ** 4 - self.x[2] ** 2 * self.x[1] ** 2),
            ]
        )
        kappa_V = 0.01
        kappa_h = np.array([0.02, 0.03])
        xi, lambda_mat = dut._calc_xi_Lambda(V=V, h=h, kappa_V=kappa_V, kappa_h=kappa_h)

        dVdx = V.Jacobian(self.x)
        dhdx = np.empty((2, self.nx), dtype=object)
        dhdx[0] = h[0].Jacobian(self.x)
        dhdx[1] = h[1].Jacobian(self.x)

        # Check xi
        assert dut.bu is not None
        xi_expected = np.empty((h.size + 1 + dut.bu.size,), dtype=object)
        xi_expected[0] = dhdx[0].dot(self.f) + kappa_h[0] * h[0]
        xi_expected[1] = dhdx[1].dot(self.f) + kappa_h[1] * h[1]
        xi_expected[2] = -dVdx.dot(self.f) - kappa_V * V
        assert dut.Au is not None
        xi_expected[-dut.Au.shape[0] :] = dut.bu

        assert xi.shape == xi_expected.shape
        for i in range(xi.size):
            if isinstance(xi[i], float):
                np.testing.assert_equal(xi[i], xi_expected[i])
            else:
                assert xi[i].CoefficientsAlmostEqual(xi_expected[i], 1e-8)

        # Check Lambda
        lambda_mat_expected = np.empty((xi_expected.size, self.nu), dtype=object)
        lambda_mat_expected[0] = -dhdx[0] @ self.g
        lambda_mat_expected[1] = -dhdx[1] @ self.g
        lambda_mat_expected[2] = dVdx @ self.g
        lambda_mat_expected[-dut.Au.shape[0] :] = dut.Au
        assert lambda_mat.shape == lambda_mat_expected.shape
        for i in range(lambda_mat.shape[0]):
            for j in range(lambda_mat.shape[1]):
                if isinstance(lambda_mat[i, j], float):
                    np.testing.assert_equal(lambda_mat[i, j], lambda_mat_expected[i, j])
                else:
                    assert lambda_mat[i, j].CoefficientsAlmostEqual(
                        lambda_mat_expected[i, j], 1e-8
                    )

    def test_search_compatible_lagrangians_w_clf_y_squared(self):
        """
        Test search_compatible_lagrangians with CLF and use_y_squared=True
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=2,
            with_clf=True,
            use_y_squared=True,
        )
        V = sym.Polynomial(dut.x[0] ** 2 + 4 * dut.x[1] ** 2 + dut.x[2] ** 2) / 0.001
        h = np.array(
            [
                sym.Polynomial(1 - dut.x[0] ** 2 - dut.x[1] ** 2),
                sym.Polynomial(2 - dut.x[0] ** 2 - dut.x[2] ** 2),
            ]
        )
        kappa_V = 0.01
        kappa_h = np.array([0.02, 0.03])
        lagrangian_degrees = mut.CompatibleLagrangianDegrees(
            lambda_y=[mut.XYDegree(x=2, y=0) for _ in range(self.nu)],
            xi_y=mut.XYDegree(x=2, y=0),
            y=None,
            y_cross=None,
            rho_minus_V=mut.XYDegree(x=4, y=2),
            h_plus_eps=[mut.XYDegree(x=4, y=2) for _ in range(dut.num_cbf)],
            state_eq_constraints=None,
        )
        barrier_eps = np.array([0.01, 0.02])

        prog, lagrangians = dut.construct_search_compatible_lagrangians(
            V, h, kappa_V, kappa_h, lagrangian_degrees, barrier_eps
        )

    def test_add_compatibility_w_clf_y_squared(self):
        """
        Test _add_compatibility with CLF and use_y_squared=True
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=2,
            with_clf=True,
            use_y_squared=True,
        )
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(dut.xy_set)

        V = sym.Polynomial(dut.x[0] ** 2 + 4 * dut.x[1] ** 2) / 0.1
        h = np.array(
            [
                sym.Polynomial(1 - dut.x[0] ** 2 - dut.x[1] ** 2),
                sym.Polynomial(2 - dut.x[0] ** 2 - dut.x[2] ** 2),
            ]
        )
        kappa_V = 0.01
        kappa_h = np.array([0.02, 0.03])

        # Set up Lagrangians.
        lambda_y_lagrangian = np.array(
            [prog.NewFreePolynomial(dut.xy_set, deg=2) for _ in range(dut.nu)]
        )
        xi_y_lagrangian = prog.NewFreePolynomial(dut.xy_set, deg=2)
        y_lagrangian = None
        y_cross_lagrangian = None
        rho_minus_V_lagrangian, _ = prog.NewSosPolynomial(dut.xy_set, degree=2)
        h_plus_eps_lagrangian = np.array(
            [prog.NewSosPolynomial(dut.xy_set, degree=2)[0] for _ in range(dut.num_cbf)]
        )
        lagrangians = mut.CompatibleLagrangians(
            lambda_y=lambda_y_lagrangian,
            xi_y=xi_y_lagrangian,
            y=y_lagrangian,
            y_cross=y_cross_lagrangian,
            rho_minus_V=rho_minus_V_lagrangian,
            h_plus_eps=h_plus_eps_lagrangian,
            state_eq_constraints=None,
        )

        xi, lambda_mat = dut._calc_xi_Lambda(V=V, h=h, kappa_V=kappa_V, kappa_h=kappa_h)

        barrier_eps = np.array([0.01, 0.02])
        poly = dut._add_compatibility(
            prog=prog,
            V=V,
            h=h,
            xi=xi,
            lambda_mat=lambda_mat,
            lagrangians=lagrangians,
            barrier_eps=barrier_eps,
            local_clf=True,
        )
        poly_expected = (
            -1
            - lagrangians.lambda_y.dot(lambda_mat.T @ dut.y_squared_poly)
            - lagrangians.xi_y * (xi.dot(dut.y_squared_poly) + 1)
            - lagrangians.rho_minus_V * (1 - V)
            - lagrangians.h_plus_eps.dot(h + barrier_eps)
        )
        assert poly.CoefficientsAlmostEqual(poly_expected, tolerance=1e-5)

    def test_add_compatibility_w_vrep1(self):
        """
        Test _add_compatibility with CLF and use_y_squared=True
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            u_vertices=np.array([[0, 1], [1, 0], [0, 0]]),
            u_extreme_rays=np.array([[-1, -1]]),
            num_cbf=2,
            with_clf=True,
            use_y_squared=True,
        )
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(dut.xy_set)

        V = sym.Polynomial(dut.x[0] ** 2 + 4 * dut.x[1] ** 2) / 0.1
        h = np.array(
            [
                sym.Polynomial(1 - dut.x[0] ** 2 - dut.x[1] ** 2),
                sym.Polynomial(2 - dut.x[0] ** 2 - dut.x[2] ** 2),
            ]
        )
        kappa_V = 0.01
        kappa_h = np.array([0.02, 0.03])

        # Set up Lagrangians.
        lagrangian_degrees = mut.CompatibleWVrepLagrangianDegrees(
            u_vertices=[mut.XYDegree(x=2, y=2) for _ in range(dut.u_vertices.shape[0])],
            u_extreme_rays=[
                mut.XYDegree(x=2, y=2) for _ in range(dut.u_extreme_rays.shape[0])
            ],
            xi_y=mut.XYDegree(x=2, y=2),
            y=None,
            y_cross=None,
            rho_minus_V=mut.XYDegree(x=2, y=2),
            h_plus_eps=[mut.XYDegree(x=2, y=4) for _ in range(h.size)],
            state_eq_constraints=None,
        )
        lagrangians = lagrangian_degrees.to_lagrangians(prog, dut.x_set, dut.y_set)

        xi, lambda_mat = dut._calc_xi_Lambda(V=V, h=h, kappa_V=kappa_V, kappa_h=kappa_h)
        barrier_eps = np.array([0.01, 0.02])
        poly = dut._add_compatibility_w_vrep(
            prog=prog,
            V=V,
            h=h,
            xi=xi,
            lambda_mat=lambda_mat,
            lagrangians=lagrangians,
            barrier_eps=barrier_eps,
            local_clf=True,
        )

        poly_expected = (
            -1
            - lagrangians.u_vertices.dot(
                -xi.dot(dut.y_squared_poly)
                + dut.y_squared_poly @ (lambda_mat @ dut.u_vertices.T)
                - 1
            )
            - lagrangians.u_extreme_rays.dot(
                dut.y_squared_poly @ lambda_mat @ dut.u_extreme_rays.T
            )
            - lagrangians.xi_y * (-xi.dot(dut.y_squared_poly) - 1)
            - lagrangians.rho_minus_V * (1 - V)
            - lagrangians.h_plus_eps.dot(h + barrier_eps)
        )

        assert poly.CoefficientsAlmostEqual(poly_expected, tolerance=1e-5)

    def test_add_compatibility_w_vrep2(self):
        """
        Test _add_compatibility with CLF and use_y_squared=False
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            u_vertices=np.array([[0, 1], [1, 0], [0, 0]]),
            u_extreme_rays=np.array([[-1, -1]]),
            num_cbf=2,
            with_clf=True,
            use_y_squared=False,
        )
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(dut.xy_set)

        V = sym.Polynomial(dut.x[0] ** 2 + 4 * dut.x[1] ** 2) / 0.1
        h = np.array(
            [
                sym.Polynomial(1 - dut.x[0] ** 2 - dut.x[1] ** 2),
                sym.Polynomial(2 - dut.x[0] ** 2 - dut.x[2] ** 2),
            ]
        )
        kappa_V = 0.01
        kappa_h = np.array([0.02, 0.03])

        # Set up Lagrangians.
        lagrangian_degrees = mut.CompatibleWVrepLagrangianDegrees(
            u_vertices=[mut.XYDegree(x=2, y=2) for _ in range(dut.u_vertices.shape[0])],
            u_extreme_rays=[
                mut.XYDegree(x=2, y=2) for _ in range(dut.u_extreme_rays.shape[0])
            ],
            xi_y=mut.XYDegree(x=2, y=2),
            y=[mut.XYDegree(x=2, y=2) for _ in range(dut.y.size)],
            y_cross=[mut.XYDegree(x=2, y=0) for _ in range(dut.y_cross_poly.size)],
            rho_minus_V=mut.XYDegree(x=2, y=2),
            h_plus_eps=[mut.XYDegree(x=2, y=4) for _ in range(h.size)],
            state_eq_constraints=None,
        )
        lagrangians = lagrangian_degrees.to_lagrangians(prog, dut.x_set, dut.y_set)

        xi, lambda_mat = dut._calc_xi_Lambda(V=V, h=h, kappa_V=kappa_V, kappa_h=kappa_h)
        barrier_eps = np.array([0.01, 0.02])
        poly = dut._add_compatibility_w_vrep(
            prog=prog,
            V=V,
            h=h,
            xi=xi,
            lambda_mat=lambda_mat,
            lagrangians=lagrangians,
            barrier_eps=barrier_eps,
            local_clf=True,
        )

        poly_expected = (
            -1
            - lagrangians.u_vertices.dot(
                -xi.dot(dut.y_poly) + dut.y_poly @ (lambda_mat @ dut.u_vertices.T) - 1
            )
            - lagrangians.u_extreme_rays.dot(
                dut.y_poly @ lambda_mat @ dut.u_extreme_rays.T
            )
            - lagrangians.xi_y * (-xi.dot(dut.y_poly) - 1)
            - lagrangians.y.dot(dut.y_poly)
            - lagrangians.y_cross.dot(dut.y_cross_poly)
            - lagrangians.rho_minus_V * (1 - V)
            - lagrangians.h_plus_eps.dot(h + barrier_eps)
        )

        assert poly.CoefficientsAlmostEqual(poly_expected, tolerance=1e-5)

    def test_add_barrier_exclude_constraint(self):
        """
        Test _add_barrier_exclude_constraint
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=2,
            with_clf=True,
            use_y_squared=True,
        )

        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x)

        exclude_set_index = 0
        h = np.array([sym.Polynomial(1 + 2 * self.x[0] * self.x[1])])
        lagrangians = mut.ExcludeRegionLagrangians(
            cbf=np.array([sym.Polynomial(1 + self.x[0])]),
            unsafe_region=np.array([sym.Polynomial(2 + self.x[0])]),
            state_eq_constraints=None,
        )

        poly = dut._add_barrier_exclude_constraint(
            prog, exclude_set_index, h, lagrangians
        )
        poly_expected = -(
            1 + lagrangians.cbf[0] * h[0]
        ) + lagrangians.unsafe_region.dot(dut.exclude_sets[exclude_set_index].l)
        assert poly.CoefficientsAlmostEqual(poly_expected, 1e-8)

    def test_certify_cbf_exclude(self):
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=1,
            with_clf=True,
            use_y_squared=True,
        )

        cbf = sym.Polynomial(1 - self.x.dot(self.x))
        exclude_region_lagrangian_degrees = mut.ExcludeRegionLagrangianDegrees(
            cbf=[2], unsafe_region=[2], state_eq_constraints=None
        )
        lagrangians = dut.certify_cbf_exclude(
            0,
            np.array([cbf]),
            exclude_region_lagrangian_degrees,
        )
        assert lagrangians is not None
        assert utils.is_sos(lagrangians.cbf[0])
        for i in range(dut.exclude_sets[0].l.size):
            assert utils.is_sos(lagrangians.unsafe_region[i])

    def test_find_max_inner_ellipsoid(self):
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=1,
            with_clf=True,
            use_y_squared=True,
        )
        S_ellipsoid = np.diag(np.array([1, 2.0, 3.0]))
        b_ellipsoid = np.array([0.5, 2, 1])
        c_ellipsoid = b_ellipsoid.dot(np.linalg.solve(S_ellipsoid, b_ellipsoid)) / 4
        V = (
            sym.Polynomial(
                self.x.dot(S_ellipsoid @ self.x) + b_ellipsoid.dot(self.x) + c_ellipsoid
            )
            / 2
        )
        h = np.array([sym.Polynomial(1 - self.x.dot(self.x))])
        S_sol, b_sol, c_sol = dut._find_max_inner_ellipsoid(
            V,
            h,
            V_contain_lagrangian_degree=utils.ContainmentLagrangianDegree(
                inner_ineq=[-1], inner_eq=[], outer=0
            ),
            h_contain_lagrangian_degree=[
                utils.ContainmentLagrangianDegree(inner_ineq=[-1], inner_eq=[], outer=0)
            ],
            x_inner_init=np.linalg.solve(S_ellipsoid, b_ellipsoid) / -2,
            max_iter=10,
            convergence_tol=1e-4,
            trust_region=100,
        )
        # Make sure the ellipsoid is within V <= 1 and h >= 0
        assert ellipsoid_utils.is_ellipsoid_contained(
            S_sol, b_sol, c_sol, S_ellipsoid, b_ellipsoid, c_ellipsoid - 2
        )
        assert ellipsoid_utils.is_ellipsoid_contained(
            S_sol, b_sol, c_sol, np.eye(3), np.zeros(3), -1
        )

    def test_add_ellipsoid_in_compatible_region_constraint(self):
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x)
        x_set = sym.Variables(self.x)
        V, _ = prog.NewSosPolynomial(x_set, 4)
        h = np.empty((2,), dtype=object)
        for i in range(h.size):
            h[i] = prog.NewFreePolynomial(x_set, 4)
        S_ellipsoid_inner = np.array([[3, 0, 1], [0, 4, 2.0], [1.0, 2.0, 4]])
        b_ellipsoid_inner = np.array([1, 3, 2])
        c_ellipsoid_inner: float = (
            b_ellipsoid_inner.dot(np.linalg.solve(S_ellipsoid_inner, b_ellipsoid_inner))
            / 4
            - 0.5
        )

        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=1,
            with_clf=True,
            use_y_squared=True,
        )

        dut._add_ellipsoid_in_compatible_region_constraint(
            prog, V, h, S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner
        )
        result = solvers.Solve(prog)
        assert result.is_success()
        # Now sample many points. If the point is in the inner ellipsoid, it
        # should also be in the compatible region.
        x_samples = np.linalg.solve(
            S_ellipsoid_inner, -b_ellipsoid_inner
        ) / 2 + np.random.random((1000, self.nx))
        in_ellipsoid = ellipsoid_utils.in_ellipsoid(
            S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner, x_samples
        )
        V_sol = result.GetSolution(V)
        h_sol = np.array([result.GetSolution(h_i) for h_i in h])
        in_compatible = dut.in_compatible_region(V_sol, h_sol, x_samples)
        assert np.all(in_compatible[in_ellipsoid])


class TestClfCbfToy:
    """
    In this test, we construct a dynamical system that we know how to compute
    compatible CLF/CBF as initial guess. This class is used for testing
    searching the CLF/CBF starting from that initial guess.
    """

    @classmethod
    def setup_class(cls):
        cls.nx = 2
        cls.nu = 1
        cls.x = sym.MakeVectorContinuousVariable(2, "x")
        cls.f = np.array(
            [
                sym.Polynomial(),
                sym.Polynomial(-cls.x[0] - 1.0 / 6 * cls.x[0] ** 3),
            ]
        )
        cls.g = np.array([[sym.Polynomial(1)], [sym.Polynomial(-1)]])

        cls.exclude_sets = [mut.ExcludeSet(np.array([sym.Polynomial(cls.x[0] + 10)]))]
        cls.within_set = None

        cls.kappa_V = 0.001
        cls.kappa_h = np.array([cls.kappa_V])
        cls.barrier_eps = np.array([0.01])

    def check_unsafe_region_by_sample(self, h: np.ndarray, x_samples):
        # Sample many points, make sure that {x | h[i] >= 0} doesn't intersect
        # with the i'th unsafe region.
        for i, exclude_set in enumerate(self.exclude_sets):
            unsafe_flag = np.all(
                np.concatenate(
                    [
                        (
                            unsafe_region_j.EvaluateIndeterminates(self.x, x_samples.T)
                            <= 0
                        ).reshape((-1, 1))
                        for unsafe_region_j in exclude_set.l
                    ],
                    axis=1,
                ),
                axis=1,
            )
            in_h = h[i].EvaluateIndeterminates(self.x, x_samples.T) >= 0
            assert np.all(np.logical_not(unsafe_flag[in_h]))

    def search_lagrangians(
        self,
    ) -> Tuple[
        mut.CompatibleClfCbf,
        mut.CompatibleLagrangians,
        mut.CompatibleLagrangianDegrees,
        mut.SafetySetLagrangians,
        mut.SafetySetLagrangianDegrees,
        sym.Polynomial,
        np.ndarray,
    ]:
        use_y_squared = True
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=1,
            with_clf=True,
            use_y_squared=use_y_squared,
        )
        V_init = sym.Polynomial(self.x[0] ** 2 + self.x[1] ** 2) / 0.01
        h_init = np.array([sym.Polynomial(0.001 - self.x[0] ** 2 - self.x[1] ** 2)])

        lagrangian_degrees = mut.CompatibleLagrangianDegrees(
            lambda_y=[mut.XYDegree(x=3, y=0)],
            xi_y=mut.XYDegree(x=2, y=0),
            y=None,
            y_cross=None,
            rho_minus_V=mut.XYDegree(x=2, y=0),
            h_plus_eps=[mut.XYDegree(x=2, y=0)],
            state_eq_constraints=None,
        )

        (
            compatible_prog,
            compatible_lagrangians,
        ) = dut.construct_search_compatible_lagrangians(
            V_init,
            h_init,
            self.kappa_V,
            self.kappa_h,
            lagrangian_degrees,
            self.barrier_eps,
        )
        solver_options = solvers.SolverOptions()
        compatible_result = solvers.Solve(compatible_prog, None, solver_options)
        assert compatible_result.is_success()

        compatible_lagrangians_result = compatible_lagrangians.get_result(
            compatible_result, coefficient_tol=1e-5
        )
        exclude_region_lagrangian_degrees = mut.ExcludeRegionLagrangianDegrees(
            cbf=[0], unsafe_region=[0], state_eq_constraints=None
        )
        safety_sets_lagrangian_degrees = mut.SafetySetLagrangianDegrees(
            exclude=[exclude_region_lagrangian_degrees], within=[]
        )

        safety_sets_lagrangians = dut.certify_cbf_safety_set(
            h=h_init,
            lagrangian_degrees=safety_sets_lagrangian_degrees,
            solver_options=None,
        )

        assert safety_sets_lagrangians is not None

        return (
            dut,
            compatible_lagrangians_result,
            lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            V_init,
            h_init,
        )

    def test_construct_search_clf_cbf_program(self):
        (
            dut,
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            _,
            _,
        ) = self.search_lagrangians()
        prog, V, h = dut._construct_search_clf_cbf_program(
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            clf_degree=2,
            cbf_degrees=[2],
            x_equilibrium=np.array([0, 0.0]),
            kappa_V=self.kappa_V,
            kappa_h=self.kappa_h,
            barrier_eps=self.barrier_eps,
        )
        solver_options = solvers.SolverOptions()
        solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 0)
        result = solvers.Solve(prog, None, solver_options)
        assert result.is_success()
        V_result = result.GetSolution(V)
        env = {self.x[i]: 0 for i in range(self.nx)}
        assert V_result.Evaluate(env) == 0
        assert sym.Monomial() not in V.monomial_to_coefficient_map()
        assert utils.is_sos(V_result)
        assert V_result.TotalDegree() == 2

        h_result = np.array([result.GetSolution(h[i]) for i in range(h.size)])
        assert all([h_result[i].TotalDegree() <= 2 for i in range(h.size)])

        # Sample many points, make sure that {x | h[i] >= 0} doesn't intersect
        # with the i'th unsafe region.
        x_samples = 10 * np.random.randn(1000, 2) - np.array([[10, 0]])
        self.check_unsafe_region_by_sample(h_result, x_samples)

    def test_search_clf_cbf_given_lagrangian_w_ellipsoid_inner(self):
        """
        Test search_clf_cbf_given_lagrangian with ellipsoid_inner
        """
        (
            dut,
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            V,
            h,
        ) = self.search_lagrangians()

        # Find the large ellipsoid inside the compatible region.
        x_equilibrium = np.array([0.0, 0.0])
        (
            S_ellipsoid_inner,
            b_ellipsoid_inner,
            c_ellipsoid_inner,
        ) = dut._find_max_inner_ellipsoid(
            V,
            h,
            V_contain_lagrangian_degree=mut.ContainmentLagrangianDegree(
                inner_ineq=[-1], inner_eq=[], outer=0
            ),
            h_contain_lagrangian_degree=[
                mut.ContainmentLagrangianDegree(inner_ineq=[-1], inner_eq=[], outer=0)
            ],
            x_inner_init=x_equilibrium,
            max_iter=10,
            convergence_tol=1e-4,
            trust_region=1000,
        )

        V_new, h_new, result = dut.search_clf_cbf_given_lagrangian(
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            clf_degree=2,
            cbf_degrees=[2],
            x_equilibrium=x_equilibrium,
            kappa_V=self.kappa_V,
            kappa_h=self.kappa_h,
            barrier_eps=self.barrier_eps,
            ellipsoid_inner=ellipsoid_utils.Ellipsoid(
                S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner
            ),
        )
        assert result.is_success()
        # Check that the compatible region contains the inner_ellipsoid.
        x_samples = np.random.randn(100, 2)
        in_ellipsoid = ellipsoid_utils.in_ellipsoid(
            S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner, x_samples
        )
        assert h_new is not None
        in_compatible = dut.in_compatible_region(V_new, h_new, x_samples)
        assert np.all(in_compatible[in_ellipsoid])

    def test_search_clf_cbf_given_lagrangian_w_compatible_states_options(self):
        (
            dut,
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            V,
            h,
        ) = self.search_lagrangians()

        compatible_states_options = mut.CompatibleStatesOptions(
            candidate_compatible_states=np.array([[0.1, 0.1], [-0.1, 0.1]]),
            anchor_states=np.array([[0, 0.0]]),
            h_anchor_bounds=[(np.array([0]), np.array([1]))],
            weight_V=1.0,
            weight_h=np.array([1.0]),
        )
        V_new, h_new, result = dut.search_clf_cbf_given_lagrangian(
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            clf_degree=2,
            cbf_degrees=[2],
            x_equilibrium=np.array([0.0, 0.0]),
            kappa_V=1e-3,
            kappa_h=np.array([1e-3]),
            barrier_eps=np.array([1e-3]),
            ellipsoid_inner=None,
            compatible_states_options=compatible_states_options,
        )
        assert V_new is not None
        assert h_new is not None
        assert result.is_success()

        # Check if bounds on h are satisfied.
        assert compatible_states_options.anchor_states is not None
        h_result_at_anchor = h_new[0].EvaluateIndeterminates(
            dut.x, compatible_states_options.anchor_states.T
        )
        assert compatible_states_options.h_anchor_bounds is not None
        assert np.all(
            h_result_at_anchor >= compatible_states_options.h_anchor_bounds[0][0]
        )
        assert np.all(
            h_result_at_anchor <= compatible_states_options.h_anchor_bounds[0][1]
        )

        print(f"V_new={V_new}")
        print(f"h_new={h_new}")

    def test_binary_search_clf_cbf_given_lagrangian(self):
        (
            dut,
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            V_init,
            h_init,
        ) = self.search_lagrangians()

        x_equilibrium = np.array([0.0, 0.0])
        (
            S_ellipsoid_inner,
            b_ellipsoid_inner,
            c_ellipsoid_inner,
        ) = dut._find_max_inner_ellipsoid(
            V_init,
            h_init,
            V_contain_lagrangian_degree=mut.ContainmentLagrangianDegree(
                inner_ineq=[-1], inner_eq=[], outer=0
            ),
            h_contain_lagrangian_degree=[
                mut.ContainmentLagrangianDegree(inner_ineq=[-1], inner_eq=[], outer=0)
            ],
            x_inner_init=x_equilibrium,
            max_iter=10,
            convergence_tol=1e-4,
            trust_region=1000,
        )

        solver_options = solvers.SolverOptions()
        solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 0)
        binary_search_scale_options = utils.BinarySearchOptions(min=1, max=50, tol=0.1)

        V, h = dut.binary_search_clf_cbf(
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            clf_degree=2,
            cbf_degrees=[2],
            x_equilibrium=x_equilibrium,
            kappa_V=self.kappa_V,
            kappa_h=self.kappa_h,
            barrier_eps=self.barrier_eps,
            ellipsoid_inner=ellipsoid_utils.Ellipsoid(
                S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner
            ),
            scale_options=binary_search_scale_options,
            solver_options=solver_options,
        )
        assert V is not None
        assert h is not None
        # Sample many points, make sure that {x | h[i] >= 0} doesn't intersect
        # with the i'th unsafe region.
        x_samples = 5 * np.random.randn(1000, 2) - np.array([[5, 0]])
        self.check_unsafe_region_by_sample(h, x_samples)

    def test_check_compatible_at_state(self):
        (
            dut,
            _,
            _,
            _,
            _,
            V_init,
            h_init,
        ) = self.search_lagrangians()
        # Since we have proved that V_init and h_init are compatible, then for
        # any state within the compatible region, there should extis control u.
        x_samples = np.random.randn(100, 2)
        in_compatible_flag = dut.in_compatible_region(V_init, h_init, x_samples)
        for i in range(x_samples.shape[0]):
            if in_compatible_flag[i]:
                is_compatible, result = dut.check_compatible_at_state(
                    V_init, h_init, x_samples[i], self.kappa_V, self.kappa_h
                )
                assert is_compatible
                assert result.is_success()


class TestClfCbfWStateEqConstraints:
    """
    Test finding CLF/CBF for system with equality constraints on the state.

    The system dynamics is
    ẋ₀ = (x₁+1)u,
    ẋ₁ = −x₀u,
    ẋ₂ = −x₀−u
    with the constraints x₀² + (x₁+1)² = 1
    """

    @classmethod
    def setup_class(cls):
        cls.nx = 3
        cls.nu = 1
        cls.x = sym.MakeVectorContinuousVariable(3, "x")
        cls.f = np.array(
            [sym.Polynomial(), sym.Polynomial(), sym.Polynomial(-cls.x[0])]
        )
        cls.g = np.array(
            [
                [sym.Polynomial(cls.x[1] + 1)],
                [sym.Polynomial(-cls.x[0])],
                [sym.Polynomial(-1)],
            ]
        )
        cls.exclude_sets = [
            mut.ExcludeSet(
                np.array([sym.Polynomial(cls.x[0] + cls.x[1] + cls.x[2] + 3)])
            )
        ]
        cls.within_set = None
        cls.state_eq_constraints = np.array(
            [sym.Polynomial(cls.x[0] ** 2 + cls.x[1] ** 2 + 2 * cls.x[1])]
        )
        cls.kappa_V = 0.001
        cls.kappa_h = np.array([cls.kappa_V])
        cls.barrier_eps = np.array([0.01])

    def search_lagrangians(self, check_result=False) -> Tuple[
        mut.CompatibleClfCbf,
        mut.CompatibleLagrangians,
        mut.CompatibleLagrangianDegrees,
        mut.SafetySetLagrangians,
        mut.SafetySetLagrangianDegrees,
        sym.Polynomial,
        np.ndarray,
    ]:
        use_y_squared = True
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=self.exclude_sets,
            within_set=self.within_set,
            Au=None,
            bu=None,
            num_cbf=1,
            with_clf=True,
            use_y_squared=use_y_squared,
            state_eq_constraints=self.state_eq_constraints,
        )
        V_init = sym.Polynomial(self.x[0] ** 2 + self.x[1] ** 2 + self.x[2] ** 2) / 0.01
        h_init = np.array(
            [sym.Polynomial(0.001 - self.x[0] ** 2 - self.x[1] ** 2 - self.x[2] ** 2)]
        )

        lagrangian_degrees = mut.CompatibleLagrangianDegrees(
            lambda_y=[mut.XYDegree(x=2, y=0)],
            xi_y=mut.XYDegree(x=2, y=0),
            y=None,
            y_cross=None,
            rho_minus_V=mut.XYDegree(x=2, y=2),
            h_plus_eps=[mut.XYDegree(x=2, y=2)],
            state_eq_constraints=[mut.XYDegree(x=2, y=2)],
        )

        (
            compatible_prog,
            compatible_lagrangians,
        ) = dut.construct_search_compatible_lagrangians(
            V_init,
            h_init,
            self.kappa_V,
            self.kappa_h,
            lagrangian_degrees,
            self.barrier_eps,
        )
        solver_options = solvers.SolverOptions()
        solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 0)
        compatible_result = solvers.Solve(compatible_prog, None, solver_options)
        assert compatible_result.is_success()
        compatible_lagrangians_result = compatible_lagrangians.get_result(
            compatible_result, coefficient_tol=None
        )

        exclude_region_lagrangian_degrees = mut.ExcludeRegionLagrangianDegrees(
            cbf=[0], unsafe_region=[0], state_eq_constraints=[0]
        )
        safety_sets_lagrangian_degrees = mut.SafetySetLagrangianDegrees(
            exclude=[exclude_region_lagrangian_degrees], within=[]
        )

        safety_sets_lagrangians = dut.certify_cbf_safety_set(
            h=h_init,
            lagrangian_degrees=safety_sets_lagrangian_degrees,
            solver_options=None,
        )

        assert safety_sets_lagrangians is not None
        if check_result:
            assert utils.is_sos(
                -1
                - safety_sets_lagrangians.exclude[0].cbf.dot(h_init)
                + safety_sets_lagrangians.exclude[0].unsafe_region.dot(
                    self.exclude_sets[0].l
                )
                - safety_sets_lagrangians.exclude[0].state_eq_constraints.dot(
                    self.state_eq_constraints
                )
            )
        return (
            dut,
            compatible_lagrangians_result,
            lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            V_init,
            h_init,
        )

    def test_search_lagrangians(self):
        self.search_lagrangians(check_result=True)

    def test_search_clf_cbf(self):
        (
            dut,
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            V_init,
            h_init,
        ) = self.search_lagrangians(check_result=False)

        (
            S_ellipsoid_inner,
            b_ellipsoid_inner,
            c_ellipsoid_inner,
        ) = dut._find_max_inner_ellipsoid(
            V_init,
            h_init,
            V_contain_lagrangian_degree=utils.ContainmentLagrangianDegree(
                inner_ineq=[-1], inner_eq=[0], outer=0
            ),
            h_contain_lagrangian_degree=[
                utils.ContainmentLagrangianDegree(
                    inner_ineq=[-1], inner_eq=[0], outer=0
                )
            ],
            x_inner_init=np.array([0, 0, 0]),
            max_iter=1,
            convergence_tol=0.1,
            trust_region=10,
        )

        V, h, result = dut.search_clf_cbf_given_lagrangian(
            compatible_lagrangians,
            compatible_lagrangian_degrees,
            safety_sets_lagrangians,
            safety_sets_lagrangian_degrees,
            clf_degree=2,
            cbf_degrees=[2],
            x_equilibrium=np.array([0.0, 0.0, 0.0]),
            kappa_V=self.kappa_V,
            kappa_h=self.kappa_h,
            barrier_eps=self.barrier_eps,
            ellipsoid_inner=ellipsoid_utils.Ellipsoid(
                S_ellipsoid_inner, b_ellipsoid_inner, c_ellipsoid_inner
            ),
        )
        assert result.is_success()
        assert V is not None
        assert sym.Monomial() not in V.monomial_to_coefficient_map().keys()
        assert h is not None


class TestCompatibleWithU:
    """
    Test the math that the polyhedron {u | Λu≤ ξ} intersects with
    u∈𝒰 = ConvexHull(u⁽¹⁾,...,u⁽ᵐ⁾) ⊕ ConvexCone(v⁽¹⁾, ..., v⁽ⁿ⁾)
    """

    def intersect(
        self,
        lambda_mat: np.ndarray,
        xi: np.ndarray,
        u_vertices: np.ndarray,
        u_extreme_rays: np.ndarray,
    ) -> bool:
        u_dim = lambda_mat.shape[1]
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(u_dim)
        prog.AddLinearConstraint(lambda_mat, np.full_like(xi, -np.inf), xi, u)
        vertices_weight = prog.NewContinuousVariables(u_vertices.shape[0])
        if vertices_weight.size > 0:
            prog.AddBoundingBoxConstraint(0, 1, vertices_weight)
            prog.AddLinearEqualityConstraint(
                np.ones((u_vertices.shape[0],)), 1, vertices_weight
            )
        extreme_rays_weight = prog.NewContinuousVariables(u_extreme_rays.shape[0])
        if extreme_rays_weight.size > 0:
            prog.AddBoundingBoxConstraint(0, np.inf, extreme_rays_weight)
        prog.AddLinearEqualityConstraint(
            u - vertices_weight @ u_vertices - extreme_rays_weight @ u_extreme_rays,
            np.zeros((u_dim,)),
        )
        result = solvers.Solve(prog)
        return result.is_success()

    def intersect_by_sos(
        self,
        lambda_mat: np.ndarray,
        xi: np.ndarray,
        u_vertices: np.ndarray,
        u_extreme_rays: np.ndarray,
        use_y_squared: bool,
        u_vertices_lagrangian_degree: List[int],
        u_extreme_rays_lagrangian_degree: List[int],
        xi_y_lagrangian_degree: Optional[int],
        y_lagrangian_degree: Optional[List[int]],
    ) -> bool:
        prog = solvers.MathematicalProgram()
        y = prog.NewIndeterminates(lambda_mat.shape[0], "y")
        y_set = sym.Variables(y)

        if use_y_squared:
            y_or_y_squared = np.array(
                [sym.Polynomial(sym.Monomial(y[i], 2)) for i in range(y.size)]
            )
        else:
            y_poly = np.array([sym.Polynomial(y[i]) for i in range(y.size)])
            y_or_y_squared = y_poly

        poly_one = sym.Polynomial(sym.Monomial())

        poly = -poly_one

        if u_vertices.shape[0] > 0:
            u_vertices_lagrangian = np.array(
                [
                    prog.NewSosPolynomial(y_set, degree)[0]
                    for degree in u_vertices_lagrangian_degree
                ]
            )
            poly -= u_vertices_lagrangian.dot(
                -xi.dot(y_or_y_squared)
                + y_or_y_squared @ (lambda_mat @ u_vertices.T)
                - poly_one
            )

        if u_extreme_rays.shape[0] > 0:
            u_extreme_rays_lagrangian = np.array(
                [
                    prog.NewSosPolynomial(y_set, degree)[0]
                    for degree in u_extreme_rays_lagrangian_degree
                ]
            )

            poly -= u_extreme_rays_lagrangian.dot(
                y_or_y_squared @ (lambda_mat @ u_extreme_rays.T)
            )
            assert xi_y_lagrangian_degree is not None
            xi_y_lagrangian = prog.NewSosPolynomial(y_set, xi_y_lagrangian_degree)[0]
            poly -= xi_y_lagrangian * (-xi.dot(y_or_y_squared) - poly_one)

        if not use_y_squared:
            assert y_lagrangian_degree is not None
            y_lagrangian = np.array(
                [
                    prog.NewSosPolynomial(y_set, degree)[0]
                    for degree in y_lagrangian_degree
                ]
            )
            poly -= y_lagrangian.dot(y_poly)

        prog.AddSosConstraint(poly)
        result = solvers.Solve(prog)
        return result.is_success()

    def intersect_tester(
        self, lambda_mat, xi, u_vertices, u_extreme_rays, intersect_expected
    ):
        assert (
            self.intersect(lambda_mat, xi, u_vertices, u_extreme_rays)
            == intersect_expected
        )
        assert (
            self.intersect_by_sos(
                lambda_mat,
                xi,
                u_vertices,
                u_extreme_rays,
                use_y_squared=True,
                u_vertices_lagrangian_degree=[2] * u_vertices.shape[0],
                u_extreme_rays_lagrangian_degree=[2] * u_extreme_rays.shape[0],
                xi_y_lagrangian_degree=None if u_extreme_rays.shape[0] == 0 else 0,
                y_lagrangian_degree=None,
            )
            == intersect_expected
        )
        assert (
            self.intersect_by_sos(
                lambda_mat,
                xi,
                u_vertices,
                u_extreme_rays,
                use_y_squared=False,
                u_vertices_lagrangian_degree=[0] * u_vertices.shape[0],
                u_extreme_rays_lagrangian_degree=[0] * u_extreme_rays.shape[0],
                xi_y_lagrangian_degree=None if u_extreme_rays.shape[0] == 0 else 0,
                y_lagrangian_degree=[0, 0, 0],
            )
            == intersect_expected
        )

    def test_convexhull_intersect(self):
        """
        The convex hull of u_vertices intersects with the polyhedron Λu≤ ξ
        """
        lambda_mat = np.array([[1, 1], [-1, 0], [0, -1]])
        xi = np.array([1, 0, 0])

        u_vertices_sequences = [
            np.array([[0.4, 0.4], [0.4, 2], [2, 0.2]]),
            np.array([[-1, -1], [0.5, 1], [1, 0.5]]),
            np.array([[1, -0.5], [-0.5, 1], [1, 1]]),
        ]
        u_extreme_rays = np.zeros((0, 2))

        intersect_expected = True
        for u_vertices in u_vertices_sequences:
            self.intersect_tester(
                lambda_mat, xi, u_vertices, u_extreme_rays, intersect_expected
            )

    def test_convexhull_not_intersect(self):
        """
        The convex hull of u_vertices does not intersect with the polyhedron Λu≤ ξ
        """
        lambda_mat = np.array([[1, 1], [-1, 0], [0, -1]])
        xi = np.array([1, 0, 0])

        u_vertices_sequences = [
            np.array([[0.55, 0.55], [0.6, 2], [2, 0.6]]),
            np.array([[-1, -1], [-0.5, -0.1], [-0.1, -0.5]]),
        ]
        u_extreme_rays = np.zeros((0, 2))

        intersect_expected = False
        for u_vertices in u_vertices_sequences:
            self.intersect_tester(
                lambda_mat, xi, u_vertices, u_extreme_rays, intersect_expected
            )

    def test_convexcone_intersect(self):
        """
        The convex cone of u_extreme_rays intersects with the polyhedron Λu≤ ξ
        """
        lambda_mat = np.array([[1, 1], [-1, 0], [0, -1]])
        xi = np.array([1, 0, 0])
        u_vertices = np.zeros((0, 2))

        u_extreme_rays = np.array([[0.1, 1], [1, 0.1]])

        self.intersect_tester(
            lambda_mat, xi, u_vertices, u_extreme_rays, intersect_expected=True
        )

        u_vertices = np.array([[-1, -1]])
        u_extreme_rays = np.array([[1, 1.1], [1.1, -1]])
        self.intersect_tester(
            lambda_mat, xi, u_vertices, u_extreme_rays, intersect_expected=True
        )

    def test_convexcone_not_intersect(self):
        """
        The convex cone of u_extreme_rays doesn't intersects with the polyhedron Λu≤ ξ
        """
        lambda_mat = np.array([[1, 1], [-1, 0], [0, -1]])
        xi = np.array([1, -0.1, -0.1])

        u_vertices = np.empty((0, 2))
        u_extreme_rays = np.array([[-1, 0], [0, -1]])
        self.intersect_tester(
            lambda_mat, xi, u_vertices, u_extreme_rays, intersect_expected=False
        )

    def test_empty_set(self):
        """
        The set Λu≤ ξ is empty.
        """
        lambda_mat = np.array([[1, 1], [-1, 0], [0, -1]])
        xi = np.array([-1, 0, 0])

        u_vertices = np.array([[0, 0], [0, 1], [1, 0]])
        u_extreme_rays = np.empty((0, 2))
        self.intersect_tester(
            lambda_mat, xi, u_vertices, u_extreme_rays, intersect_expected=False
        )
