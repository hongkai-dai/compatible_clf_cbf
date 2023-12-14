import compatible_clf_cbf.clf_cbf as mut

from typing import List, Tuple

import numpy as np
import pytest  # noqa

import pydrake.solvers as solvers
import pydrake.symbolic as sym

import compatible_clf_cbf.ellipsoid_utils as ellipsoid_utils
import compatible_clf_cbf.utils as utils


class TestCompatibleLagrangianDegrees(object):
    def test_construct_polynomial(self):
        degree = mut.CompatibleLagrangianDegrees.Degree(x=3, y=2)
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


class TestClfCbf(object):
    @classmethod
    def setup_class(cls):
        cls.nx = 3
        cls.nu = 2
        cls.x = sym.MakeVectorContinuousVariable(cls.nx, "x")
        # The dynamics and unsafe regions are arbitrary, they are only used to
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
        cls.unsafe_regions = [
            np.array([sym.Polynomial(cls.x[0] + 1)]),
            np.array([sym.Polynomial(1 - cls.x[1]), sym.Polynomial(1 - cls.x[0])]),
        ]

    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the dynamics at x=0 at u = 0
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
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
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

        assert dut.y.shape == (len(self.unsafe_regions) + 1,)
        check_members(dut)

        # Now construct with with_clf=False
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=False,
            use_y_squared=True,
        )
        assert dut.y.shape == (len(self.unsafe_regions),)
        check_members(dut)

    def test_calc_xi_Lambda_w_clf(self):
        """
        Test _calc_xi_Lambda with CLF.
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=True,
            use_y_squared=True,
        )
        V = sym.Polynomial(
            self.x[0] ** 2 + self.x[1] ** 2 + self.x[2] ** 2 + self.x[0] * 2
        )
        b = np.array(
            [
                sym.Polynomial(1 - self.x[0] ** 2 - self.x[1] ** 2 - self.x[2] ** 2),
                sym.Polynomial(2 - self.x[0] ** 4 - self.x[2] ** 2 * self.x[1] ** 2),
            ]
        )
        kappa_V = 0.01
        kappa_b = np.array([0.02, 0.03])
        xi, lambda_mat = dut._calc_xi_Lambda(V=V, b=b, kappa_V=kappa_V, kappa_b=kappa_b)
        assert xi.shape == (1 + len(self.unsafe_regions),)
        assert lambda_mat.shape == (1 + len(self.unsafe_regions), dut.nu)
        dbdx = np.empty((2, 3), dtype=object)
        dbdx[0] = b[0].Jacobian(self.x)
        dbdx[1] = b[1].Jacobian(self.x)
        dVdx = V.Jacobian(self.x)
        xi_expected = np.empty((3,), dtype=object)
        xi_expected[0] = dbdx[0] @ self.f + kappa_b[0] * b[0]
        xi_expected[1] = dbdx[1] @ self.f + kappa_b[1] * b[1]
        xi_expected[2] = -dVdx @ self.f - kappa_V * V
        utils.check_polynomial_arrays_equal(xi, xi_expected, 1e-8)

        lambda_mat_expected = np.empty((3, self.nu), dtype=object)
        lambda_mat_expected[:2] = -dbdx @ self.g
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
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=False,
            use_y_squared=True,
        )
        V = None
        b = np.array(
            [
                sym.Polynomial(1 - self.x[0] ** 2 - self.x[1] ** 2 - self.x[2] ** 2),
                sym.Polynomial(2 - self.x[0] ** 4 - self.x[2] ** 2 * self.x[1] ** 2),
            ]
        )
        kappa_V = None
        kappa_b = np.array([0.02, 0.03])
        xi, lambda_mat = dut._calc_xi_Lambda(V=V, b=b, kappa_V=kappa_V, kappa_b=kappa_b)
        assert xi.shape == (len(self.unsafe_regions),)
        assert lambda_mat.shape == (len(self.unsafe_regions), dut.nu)
        dbdx = np.empty((2, 3), dtype=object)
        dbdx[0] = b[0].Jacobian(self.x)
        dbdx[1] = b[1].Jacobian(self.x)
        xi_expected = np.empty((2,), dtype=object)
        xi_expected[0] = dbdx[0] @ self.f + kappa_b[0] * b[0]
        xi_expected[1] = dbdx[1] @ self.f + kappa_b[1] * b[1]
        utils.check_polynomial_arrays_equal(xi, xi_expected, 1e-8)

        lambda_mat_expected = np.empty((2, self.nu), dtype=object)
        lambda_mat_expected = -dbdx @ self.g
        utils.check_polynomial_arrays_equal(lambda_mat, lambda_mat_expected, 1e-8)

    def test_search_compatible_lagrangians_w_clf_y_squared(self):
        """
        Test search_compatible_lagrangians with CLF and use_y_squared=True
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=True,
            use_y_squared=True,
        )
        V = sym.Polynomial(dut.x[0] ** 2 + 4 * dut.x[1] ** 2 + dut.x[2] ** 2)
        b = np.array(
            [
                sym.Polynomial(1 - dut.x[0] ** 2 - dut.x[1] ** 2),
                sym.Polynomial(2 - dut.x[0] ** 2 - dut.x[2] ** 2),
            ]
        )
        kappa_V = 0.01
        kappa_b = np.array([0.02, 0.03])
        lagrangian_degrees = mut.CompatibleLagrangianDegrees(
            lambda_y=[
                mut.CompatibleLagrangianDegrees.Degree(x=2, y=0) for _ in range(self.nu)
            ],
            xi_y=mut.CompatibleLagrangianDegrees.Degree(x=2, y=0),
            y=None,
            rho_minus_V=mut.CompatibleLagrangianDegrees.Degree(x=4, y=2),
            b_plus_eps=[
                mut.CompatibleLagrangianDegrees.Degree(x=4, y=2)
                for _ in range(len(self.unsafe_regions))
            ],
        )
        rho = 0.001
        barrier_eps = np.array([0.01, 0.02])

        prog, lagrangians = dut.construct_search_compatible_lagrangians(
            V, b, kappa_V, kappa_b, lagrangian_degrees, rho, barrier_eps
        )

    def test_add_compatibility_w_clf_y_squared(self):
        """
        Test _add_compatibility with CLF and use_y_squared=True
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=True,
            use_y_squared=True,
        )
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(dut.xy_set)

        V = sym.Polynomial(dut.x[0] ** 2 + 4 * dut.x[1] ** 2)
        b = np.array(
            [
                sym.Polynomial(1 - dut.x[0] ** 2 - dut.x[1] ** 2),
                sym.Polynomial(2 - dut.x[0] ** 2 - dut.x[2] ** 2),
            ]
        )
        kappa_V = 0.01
        kappa_b = np.array([0.02, 0.03])

        # Set up Lagrangians.
        lambda_y_lagrangian = np.array(
            [prog.NewFreePolynomial(dut.xy_set, deg=2) for _ in range(dut.nu)]
        )
        xi_y_lagrangian = prog.NewFreePolynomial(dut.xy_set, deg=2)
        y_lagrangian = None
        rho_minus_V_lagrangian, _ = prog.NewSosPolynomial(dut.xy_set, degree=2)
        b_plus_eps_lagrangian = np.array(
            [
                prog.NewSosPolynomial(dut.xy_set, degree=2)[0]
                for _ in range(len(dut.unsafe_regions))
            ]
        )
        lagrangians = mut.CompatibleLagrangians(
            lambda_y=lambda_y_lagrangian,
            xi_y=xi_y_lagrangian,
            y=y_lagrangian,
            rho_minus_V=rho_minus_V_lagrangian,
            b_plus_eps=b_plus_eps_lagrangian,
        )

        rho = 0.1
        barrier_eps = np.array([0.01, 0.02])
        poly = dut._add_compatibility(
            prog=prog,
            V=V,
            b=b,
            kappa_V=kappa_V,
            kappa_b=kappa_b,
            lagrangians=lagrangians,
            rho=rho,
            barrier_eps=barrier_eps,
        )
        (xi, lambda_mat) = dut._calc_xi_Lambda(
            V=V, b=b, kappa_V=kappa_V, kappa_b=kappa_b
        )
        poly_expected = (
            -1
            - lagrangians.lambda_y.dot(lambda_mat.T @ dut.y_squared_poly)
            - lagrangians.xi_y * (xi.dot(dut.y_squared_poly) + 1)
            - lagrangians.rho_minus_V * (rho - V)
            - lagrangians.b_plus_eps.dot(b + barrier_eps)
        )
        assert poly.CoefficientsAlmostEqual(poly_expected, tolerance=1e-5)

    def test_add_barrier_safe_constraint(self):
        """
        Test _add_barrier_safe_constraint
        """
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=True,
            use_y_squared=True,
        )

        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.x)

        unsafe_region_index = 0
        b = sym.Polynomial(1 + 2 * self.x[0] * self.x[1])
        lagrangians = mut.UnsafeRegionLagrangians(
            cbf=sym.Polynomial(1 + self.x[0]),
            unsafe_region=np.array([sym.Polynomial(2 + self.x[0])]),
        )

        poly = dut._add_barrier_safe_constraint(
            prog, unsafe_region_index, b, lagrangians
        )
        poly_expected = -(1 + lagrangians.cbf) * b + lagrangians.unsafe_region.dot(
            dut.unsafe_regions[unsafe_region_index]
        )
        assert poly.CoefficientsAlmostEqual(poly_expected, 1e-8)

    def test_certify_cbf_unsafe_region(self):
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=True,
            use_y_squared=True,
        )

        cbf = sym.Polynomial(1 - self.x.dot(self.x))
        lagrangians = dut.certify_cbf_unsafe_region(
            0, cbf, cbf_lagrangian_degree=2, unsafe_region_lagrangian_degrees=[2]
        )
        assert utils.is_sos(lagrangians.cbf)
        for i in range(dut.unsafe_regions[0].size):
            assert utils.is_sos(lagrangians.unsafe_region[i])

    def test_find_max_inner_ellipsoid(self):
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=True,
            use_y_squared=True,
        )
        S_ellipsoid = np.diag(np.array([1, 2.0, 3.0]))
        b_ellipsoid = np.array([0.5, 2, 1])
        c_ellipsoid = b_ellipsoid.dot(np.linalg.solve(S_ellipsoid, b_ellipsoid)) / 4
        V = sym.Polynomial(
            self.x.dot(S_ellipsoid @ self.x) + b_ellipsoid.dot(self.x) + c_ellipsoid
        )
        b = np.array([sym.Polynomial(1 - self.x.dot(self.x))])
        rho = 2
        S_sol, b_sol, c_sol = dut._find_max_inner_ellipsoid(
            V,
            b,
            rho,
            V_contain_lagrangian_degree=utils.ContainmentLagrangianDegree(
                inner=-1, outer=0
            ),
            b_contain_lagrangian_degree=[
                utils.ContainmentLagrangianDegree(inner=-1, outer=0)
            ],
            x_inner_init=np.linalg.solve(S_ellipsoid, b_ellipsoid) / -2,
            max_iter=10,
            convergence_tol=1e-4,
            trust_region=100,
        )
        # Make sure the ellipsoid is within V<= rho and b >= 0
        assert ellipsoid_utils.is_ellipsoid_contained(
            S_sol, b_sol, c_sol, S_ellipsoid, b_ellipsoid, c_ellipsoid - rho
        )
        assert ellipsoid_utils.is_ellipsoid_contained(
            S_sol, b_sol, c_sol, np.eye(3), np.zeros(3), -1
        )


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

        cls.unsafe_regions = [np.array([sym.Polynomial(cls.x[0] + 10)])]

        cls.kappa_V = 0.001
        cls.kappa_b = np.array([cls.kappa_V])
        cls.barrier_eps = np.array([0.01])

    def search_lagrangians(
        self,
    ) -> Tuple[
        mut.CompatibleClfCbf,
        mut.CompatibleLagrangians,
        List[mut.UnsafeRegionLagrangians],
    ]:
        use_y_squared = True
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=True,
            use_y_squared=use_y_squared,
        )
        V_init = sym.Polynomial(self.x[0] ** 2 + self.x[1] ** 2)
        b_init = np.array([sym.Polynomial(0.001 - self.x[0] ** 2 - self.x[1] ** 2)])

        lagrangian_degrees = mut.CompatibleLagrangianDegrees(
            lambda_y=[mut.CompatibleLagrangianDegrees.Degree(x=3, y=0)],
            xi_y=mut.CompatibleLagrangianDegrees.Degree(x=0, y=0),
            y=None,
            rho_minus_V=mut.CompatibleLagrangianDegrees.Degree(x=2, y=0),
            b_plus_eps=[mut.CompatibleLagrangianDegrees.Degree(x=2, y=0)],
        )
        rho = 0.01

        (
            compatible_prog,
            compatible_lagrangians,
        ) = dut.construct_search_compatible_lagrangians(
            V_init,
            b_init,
            self.kappa_V,
            self.kappa_b,
            lagrangian_degrees,
            rho,
            self.barrier_eps,
        )
        solver_options = solvers.SolverOptions()
        solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 1)
        solver = solvers.ClarabelSolver()
        compatible_result = solver.Solve(compatible_prog, None, solver_options)
        assert compatible_result.is_success()

        compatible_lagrangians_result = compatible_lagrangians.get_result(
            compatible_result, coefficient_tol=1e-5
        )

        unsafe_lagrangians = [
            dut.certify_cbf_unsafe_region(
                unsafe_region_index=0,
                cbf=b_init[0],
                cbf_lagrangian_degree=0,
                unsafe_region_lagrangian_degrees=[0],
                solver_options=None,
            )
        ]

        return dut, compatible_lagrangians_result, unsafe_lagrangians

    def test_search_clf_cbf(self):
        (
            dut,
            compatible_lagrangians,
            unsafe_lagrangians,
        ) = self.search_lagrangians()
        prog, V, b, rho = dut._construct_search_clf_cbf_program(
            compatible_lagrangians,
            unsafe_lagrangians,
            clf_degree=2,
            cbf_degrees=[2],
            x_equilibrium=np.array([0, 0.0]),
            kappa_V=self.kappa_V,
            kappa_b=self.kappa_b,
            barrier_eps=self.barrier_eps,
        )
        solver_options = solvers.SolverOptions()
        solver_options.SetOption(solvers.CommonSolverOption.kPrintToConsole, 1)
        solver = solvers.ClarabelSolver()
        result = solver.Solve(prog, None, solver_options)
        assert result.is_success()
        V_result = result.GetSolution(V)
        env = {self.x[i]: 0 for i in range(self.nx)}
        assert V_result.Evaluate(env) == 0
        assert sym.Monomial() not in V.monomial_to_coefficient_map()
        assert utils.is_sos(V_result, solvers.ClarabelSolver.id())
        assert V_result.TotalDegree() == 2
        rho_result = result.GetSolution(rho)
        assert rho_result >= 0

        b_result = np.array([result.GetSolution(b[i]) for i in range(b.size)])
        assert all([b_result[i].TotalDegree() <= 2 for i in range(b.size)])
