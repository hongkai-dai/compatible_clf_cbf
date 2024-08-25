from typing import Optional

import numpy as np

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.framework

import compatible_clf_cbf.clf
import compatible_clf_cbf.cbf

import compatible_clf_cbf.utils


class ClfCbfController(pydrake.systems.framework.LeafSystem):
    def __init__(
        self,
        f: np.ndarray,
        g: np.ndarray,
        V: sym.Polynomial,
        b: np.ndarray,
        x: np.ndarray,
        kappa_V: float,
        kappa_b: np.ndarray,
        Qu: np.ndarray,
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray],
        solver_id: Optional[solvers.SolverId],
        solver_options: Optional[solvers.SolverOptions],
    ):
        super().__init__()
        self.nu = g.shape[1]
        self.DeclareVectorInputPort("state", x.size)
        self.action_output_index = self.DeclareVectorOutputPort(
            "action", self.nu, self.calc_action
        ).get_index()
        self.V = V
        self.b = b
        self.x = x
        self.V_output_index = self.DeclareVectorOutputPort(
            "V", 1, self.calc_V
        ).get_index()
        self.b_output_index = self.DeclareVectorOutputPort(
            "b", b.size, self.calc_b
        ).get_index()
        self.clf_constraint = compatible_clf_cbf.clf.ClfConstraint(V, f, g, x, kappa_V)
        self.cbf_constraint = [
            compatible_clf_cbf.cbf.CbfConstraint(b[i], f, g, x, kappa_b[i])
            for i in range(b.size)
        ]
        self.Qu = Qu
        self.Au = Au
        self.bu = bu
        self.solver_id = solver_id
        self.solver_options = solver_options

    def action_output_port(self):
        return self.get_output_port(self.action_output_index)

    def clf_output_port(self):
        return self.get_output_port(self.V_output_index)

    def cbf_output_port(self):
        return self.get_output_port(self.b_output_index)

    def calc_action(self, context: pydrake.systems.framework.Context, output):
        x_val: np.ndarray = self.get_input_port(0).Eval(context)
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(self.nu, "u")
        prog.AddQuadraticCost(self.Qu, np.zeros((self.nu,)), u, is_convex=True)
        self.clf_constraint.add_to_prog(prog, x_val, u)
        for cbf_cnstr in self.cbf_constraint:
            cbf_cnstr.add_to_prog(prog, x_val, u)
        if self.Au is not None:
            assert self.bu is not None
            prog.AddLinearConstraint(
                self.Au, np.full_like(self.bu, -np.inf), self.bu, u
            )
        result = compatible_clf_cbf.utils.solve_with_id(
            prog, self.solver_id, self.solver_options
        )
        assert result.is_success()
        u_val = result.GetSolution(u)
        output.set_value(u_val)

    def calc_V(self, context: pydrake.systems.framework.Context, output):
        x_val: np.ndarray = self.get_input_port(0).Eval(context)
        env = {self.x[i]: x_val[i] for i in range(self.x.size)}
        V_val = self.V.Evaluate(env)
        output.set_value(np.array([V_val]))

    def calc_b(self, context: pydrake.systems.framework.Context, output):
        x_val: np.ndarray = self.get_input_port(0).Eval(context)
        env = {self.x[i]: x_val[i] for i in range(self.x.size)}
        b_val = np.array([b_i.Evaluate(env) for b_i in self.b])
        output.set_value(b_val)
