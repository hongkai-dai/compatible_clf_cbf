import compatible_clf_cbf.clf_cbf as mut

import numpy as np
import pytest  # noqa

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
        assert dut.y.shape == (len(self.unsafe_regions) + 1,)

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
