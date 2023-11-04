import compatible_clf_cbf.clf_cbf as mut

import numpy as np
import pytest  # noqa

import pydrake.solvers as solvers
import pydrake.symbolic as sym

import compatible_clf_cbf.utils as utils


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
                sym.Polynomial(cls.x[0] * cls.x[0]),
                sym.Polynomial(cls.x[1] * cls.x[0]),
                sym.Polynomial(cls.x[0] + cls.x[2]),
            ]
        )
        cls.g = np.array(
            [
                [sym.Polynomial(cls.x[0] + cls.x[1]), sym.Polynomial()],
                [sym.Polynomial(), sym.Polynomial(cls.x[1] * cls.x[2])],
                [sym.Polynomial(cls.x[0] + cls.x[2]), sym.Polynomial(cls.x[1])],
            ]
        )
        cls.unsafe_regions = [
            np.array([sym.Polynomial(cls.x[0] + 1)]),
            np.array([sym.Polynomial(1 - cls.x[1]), sym.Polynomial(1 - cls.x[0])]),
        ]

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

        V = prog.NewFreePolynomial(dut.x_set, deg=2)
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
