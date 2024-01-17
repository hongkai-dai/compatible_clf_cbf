import compatible_clf_cbf.cbf as mut

from typing import Tuple

import numpy as np
import pytest  # noqa

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers

from compatible_clf_cbf.clf_cbf import UnsafeRegionLagrangians
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

    def get_b_init(self) -> sym.Polynomial:
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
            unsafe_region=np.array(
                [
                    sym.Polynomial(self.x[0] + self.x[1] + self.x[2] + 2),
                    sym.Polynomial(2 - self.x[0] - self.x[1] - self.x[2]),
                ]
            ),
            u_vertices=None,
            state_eq_constraints=None,
        )
        b = self.get_b_init()
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(dut.x_set)

        lagrangians = UnsafeRegionLagrangians(
            cbf=sym.Polynomial(1 + self.x[0]),
            unsafe_region=np.array(
                [sym.Polynomial(2 + self.x[0]), sym.Polynomial(3 - self.x[1])]
            ),
            state_eq_constraints=None,
        )

        poly = dut._add_barrier_safe_constraint(prog, b, lagrangians)
        poly_expected = -(1 + lagrangians.cbf) * b + lagrangians.unsafe_region.dot(
            dut.unsafe_region
        )
        assert poly.CoefficientsAlmostEqual(poly_expected, 1e-8)

    def test_add_cbf_condition(self):
        dut = mut.ControlBarrier(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_region=np.array(
                [
                    sym.Polynomial(self.x[0] + self.x[1] + self.x[2] + 2),
                    sym.Polynomial(2 - self.x[0] - self.x[1] - self.x[2]),
                ]
            ),
            u_vertices=None,
            state_eq_constraints=None,
        )
        b_init = self.get_b_init()
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(dut.x_set)
        lagrangian_degrees = mut.CbfWoInputLimitLagrangianDegrees(
            dbdx_times_f=2, dbdx_times_g=[2, 2], b_plus_eps=2, state_eq_constraints=None
        )
        lagrangians = lagrangian_degrees.to_lagrangians(prog, dut.x_set)
        eps = 0.01
        kappa = 0.001
        sos_poly = dut._add_cbf_condition(prog, b_init, lagrangians, eps, kappa)
        result = solvers.Solve(prog)
        assert result.is_success()

        lagrangians_result = lagrangians.get_result(result, coefficient_tol=None)
        assert is_sos(lagrangians_result.dbdx_times_f)

        dbdx = b_init.Jacobian(self.x)
        dbdx_times_f = dbdx.dot(self.f)
        dbdx_times_g = dbdx @ self.g

        sos_poly_expected = (
            (1 + lagrangians.dbdx_times_f) * (dbdx_times_f + kappa * b_init)
            - lagrangians.dbdx_times_g.dot(dbdx_times_g)
            - lagrangians.b_plus_eps * (b_init + eps)
        )
        sos_poly_result = result.GetSolution(sos_poly)
        sos_poly_expected_result = result.GetSolution(sos_poly_expected)
        assert sos_poly_result.CoefficientsAlmostEqual(sos_poly_expected_result, 1e-5)
