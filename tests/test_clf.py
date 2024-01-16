import compatible_clf_cbf.clf as mut

from typing import Tuple

import numpy as np
import pytest  # noqa

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers


class TestClf:
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

    def get_V_init(self) -> sym.Polynomial:
        A, B = self.linearize()
        K, S = controllers.LinearQuadraticRegulator(
            A, B, Q=np.eye(self.nx), R=np.eye(self.nu)
        )
        return sym.Polynomial(self.x.dot(S @ self.x))

    def test_add_clf_condition_wo_input_limit(self):
        dut = mut.ControlLyapunov(
            f=self.f,
            g=self.g,
            x=self.x,
            x_equilibrium=np.zeros(self.nx),
            u_vertices=None,
        )
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(dut.x_set)
        V = self.get_V_init()

        lagrangian_degrees = mut.ClfWoInputLimitLagrangianDegrees(
            dVdx_times_f=2,
            dVdx_times_g=[4, 4],
            rho_minus_V=4,
            state_eq_constraints=None,
        )

        lagrangians = lagrangian_degrees.to_lagrangians(
            prog, dut.x_set, dut.x_equilibrium
        )

        # The Lagrangian for rho-V doesn't contain constant or linear terms.
        assert (
            sym.Monomial()
            not in lagrangians.rho_minus_V.monomial_to_coefficient_map().keys()
        )
        for x_i in dut.x_set:
            assert (
                sym.Monomial(x_i)
                not in lagrangians.rho_minus_V.monomial_to_coefficient_map().keys()
            )

        rho = 0.1
        kappa = 0.01

        condition = dut._add_clf_condition(prog, V, lagrangians, rho, kappa)
        dVdx = V.Jacobian(dut.x)
        dVdx_times_f = dVdx.dot(dut.f)
        dVdx_times_g = (dVdx @ dut.g).reshape((-1,))

        condition_expected = (
            -(1 + lagrangians.dVdx_times_f) * (dVdx_times_f + kappa * V)
            - lagrangians.dVdx_times_g.dot(dVdx_times_g)
            - lagrangians.rho_minus_V * (rho - V)
        )
        assert condition.EqualTo(condition_expected)

    def test_search_lagrangian_given_clf_wo_input_limit(self):
        """
        Test search_lagrangian_given_clf without input limits.
        """
        dut = mut.ControlLyapunov(
            f=self.f,
            g=self.g,
            x=self.x,
            x_equilibrium=np.zeros(self.nx),
            u_vertices=None,
        )
        V = self.get_V_init()

        lagrangians = dut.search_lagrangian_given_clf(
            V,
            rho=0.01,
            kappa=0.001,
            lagrangian_degrees=mut.ClfWoInputLimitLagrangianDegrees(
                dVdx_times_f=2,
                dVdx_times_g=[3, 3],
                rho_minus_V=4,
                state_eq_constraints=None,
            ),
        )
        assert isinstance(lagrangians, mut.ClfWoInputLimitLagrangian)
