import compatible_clf_cbf.cbf as mut

from typing import Tuple

import numpy as np
import pytest  # noqa

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers

from compatible_clf_cbf.clf_cbf import ExcludeRegionLagrangians
import compatible_clf_cbf.clf_cbf as clf_cbf
from compatible_clf_cbf.utils import is_sos


class TestCbf:
    @classmethod
    def setup_class(cls):
        cls.nx = 3
        cls.nu = 2
        cls.x = sym.MakeVectorContinuousVariable(cls.nx, "x")
        cls.f = np.array(
            [
                sym.Polynomial(-cls.x[0]),
                sym.Polynomial(cls.x[0] * cls.x[1] * cls.x[2] - cls.x[1]),
                sym.Polynomial(cls.x[1] ** 3 - 2 * cls.x[0]),
            ]
        )
        cls.g = np.array(
            [
                [sym.Polynomial(1), sym.Polynomial(cls.x[0] ** 2)],
                [sym.Polynomial(cls.x[0] * cls.x[2]), sym.Polynomial(1)],
                [sym.Polynomial(cls.x[1] * cls.x[2]), sym.Polynomial(3)],
            ]
        )
        # for high relative degree test:
        # this will be used to test the HOCBF constraint calculation.
        cls.high_f = np.array([
            sym.Polynomial(cls.x[1]),
            sym.Polynomial(cls.x[2]),
            sym.Polynomial(0)
        ])
        cls.high_g = np.array([
            [sym.Polynomial(0)],
            [sym.Polynomial(0)],
            [sym.Polynomial(1)]
        ])

    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the dynamics at x=0 and u=0.
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

    def get_h_init(self) -> sym.Polynomial:
        A, B = self.linearize()
        K, S = controllers.LinearQuadraticRegulator(
            A, B, Q=np.eye(self.nx), R=np.eye(self.nu)
        )
        return 0.1 - sym.Polynomial(self.x.dot(S @ self.x))

    def test_add_barrier_safe_constraint(self):
        dut = mut.ControlBarrier(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=[
                clf_cbf.ExcludeSet(
                    np.array(
                        [
                            sym.Polynomial(self.x[0] + self.x[1] + self.x[2] + 2),
                            sym.Polynomial(2 - self.x[0] - self.x[1] - self.x[2]),
                        ]
                    ),
                )
            ],
            within_set=None,
            u_vertices=None,
            state_eq_constraints=None,
        )
        h = self.get_h_init()
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(dut.x_set)

        lagrangians = ExcludeRegionLagrangians(
            cbf=np.array([sym.Polynomial(1 + self.x[0])]),
            unsafe_region=np.array(
                [sym.Polynomial(2 + self.x[0]), sym.Polynomial(3 - self.x[1])]
            ),
            state_eq_constraints=None,
        )

        exclude_set_index = 0
        poly = dut._add_barrier_exclude_constraint(
            prog, exclude_set_index, h, lagrangians
        )
        poly_expected = -(1 + lagrangians.cbf[0]) * h + lagrangians.unsafe_region.dot(
            dut.exclude_sets[exclude_set_index].l
        )
        assert poly.CoefficientsAlmostEqual(poly_expected, 1e-8)

    def test_add_cbf_derivative_condition(self):
        dut = mut.ControlBarrier(
            f=self.f,
            g=self.g,
            x=self.x,
            exclude_sets=[
                clf_cbf.ExcludeSet(
                    np.array(
                        [
                            sym.Polynomial(self.x[0] + self.x[1] + self.x[2] + 2),
                            sym.Polynomial(2 - self.x[0] - self.x[1] - self.x[2]),
                        ]
                    )
                )
            ],
            within_set=None,
            u_vertices=None,
            state_eq_constraints=None,
        )
        h_init = self.get_h_init()
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(dut.x_set)
        lagrangian_degrees = mut.CbfWoInputLimitLagrangianDegrees(
            dhdx_times_f=2, dhdx_times_g=[2, 2], h_plus_eps=2, state_eq_constraints=None
        )
        lagrangians = lagrangian_degrees.to_lagrangians(prog, dut.x_set)
        eps = 0.01
        kappa = 0.001
        sos_poly = dut._add_cbf_derivative_condition(
            prog, h_init, lagrangians, eps, kappa
        )
        result = solvers.Solve(prog)
        assert result.is_success()

        lagrangians_result = lagrangians.get_result(result, coefficient_tol=None)
        assert is_sos(lagrangians_result.dhdx_times_f)

        dhdx = h_init.Jacobian(self.x)
        dhdx_times_f = dhdx.dot(self.f)
        dhdx_times_g = dhdx @ self.g

        sos_poly_expected = (
            (1 + lagrangians.dhdx_times_f) * (dhdx_times_f + kappa * h_init)
            - lagrangians.dhdx_times_g.dot(dhdx_times_g)
            - lagrangians.h_plus_eps * (h_init + eps)
        )
        sos_poly_result = result.GetSolution(sos_poly)
        sos_poly_expected_result = result.GetSolution(sos_poly_expected)
        assert sos_poly_result.CoefficientsAlmostEqual(sos_poly_expected_result, 1e-5)

    def test_hocbf_control_constraint(self):
        """
        This function tests the calculation of hocbf
        control constraint.
        """
        hx = sym.Polynomial(1 - self.x[0]**2)
        kappa_1 = 1
        kappa_2 = 1
        kappa_3 = 1
        test_object = mut.CbfConstraint(
            h=hx,
            f=self.high_f,
            g=self.high_g,
            x=self.x,
            kappa=[kappa_1, kappa_2, kappa_3]
        )
        dh_dx = hx.Jacobian(self.x)
        Lf_hx = dh_dx.dot(self.high_f)
        Lf_2_hx = Lf_hx.Jacobian(self.x).dot(self.high_f)
        Lf_3_hx = Lf_2_hx.Jacobian(self.x).dot(self.high_f)
        Lf_2_Lg_hx = Lf_2_hx.Jacobian(self.x).dot(self.high_g)
        lie_derivatives_vec = np.array([Lf_3_hx, Lf_2_hx, Lf_hx, hx])
        kappa_elements_vec = np.array([
            1,
            kappa_1+kappa_2+kappa_3,
            kappa_1*kappa_2+kappa_1*kappa_3+kappa_2*kappa_3,
            kappa_1*kappa_2*kappa_3
        ])
        expected_lhs_coeff = Lf_2_Lg_hx
        expected_rhs = -lie_derivatives_vec.dot(kappa_elements_vec)

        assert expected_lhs_coeff.shape == test_object.lhs_coeff.shape
        for i in range(expected_lhs_coeff.shape[0]):
            assert expected_lhs_coeff[i].CoefficientsAlmostEqual(
                test_object.lhs_coeff[i], 1e-8
            )
        assert expected_rhs.CoefficientsAlmostEqual(
            test_object.rhs, 1e-8
        )
