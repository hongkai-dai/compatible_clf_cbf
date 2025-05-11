from typing import Tuple
import numpy as np
import pydrake.symbolic as sym
import pydrake.systems.framework

"""
This example is about inverted pendulum. Since the original dynamics has sin() function,
In this example we will use the trig-poly system.
Orginially:
x0_dot = x1;
x1_dot = g/l*sin(x0) + (1/ml^2)*u

define: z0 = sin(x0), z1 = cos(x0)-1, z2 = x1
z0_dot = (z1+1)*z2,
z1_dot = -z0*z2,
z2_dot = g/l*z0 + 1/ml^2*u
also, state_eq_cosnt: z0^2 + (z1 + 1)^2 - 1 = 0
"""


class InvertedPendulumPlant(pydrake.systems.framework.LeafSystem):
    """
    This class defines the inverted pendulum system. It is inherited
    from LeafSystem, meaning that we will use it for both the CLF-CBF
    verfication and synthesis, and also the simulation of CLF-CBF-QP
    as well.
    The initialization function defines the system parameters and
    also the input, state and output ports.
    The function DoCalcTimeDerivatives and trig_poly_dynamics
    defines the dynamics of the system. There functions are used for
    simulation.
    For verification and synthesis, we use the function
    trig_poly_affine_dynamics, trig_poly_state_eq_const.
    The function trig_poly_linearized_dynamics is used for initializing
    the CLF using LQR.
    """
    m: float  # mass
    l: float  # length
    g: float  # gravity constant

    def __init__(self):
        super().__init__()
        self.m = 2
        self.l = 0.5
        self.g = 9.81
        self.DeclareVectorInputPort(name="u", size=1)
        state_index = self.DeclareContinuousState(num_state_variables=3)
        self.DeclareStateOutputPort(name="x", state_index=state_index)

    def DoCalcTimeDerivatives(
        self,
        context: pydrake.systems.framework.Context,
        derivatives: pydrake.systems.framework.ContinuousState,
    ):
        x = context.get_continuous_state_vector().CopyToVector()
        u = self.EvalVectorInput(context, 0).CopyToVector()
        xdot: np.ndarray = self.trig_poly_dynamics(x, u)
        derivatives.SetFromVector(xdot)

    def trig_poly_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple:
        """
        This should be used for the simulations of inverted pendulum.
        """
        assert x.shape == (3,)
        f = np.array([(x[1] + 1) * x[2], -x[0] * x[2], (self.g / self.l) * x[0]])
        g = np.array([[0], [0], [1 / (self.m * self.l**2)]])
        xDt = f + np.dot(g, u)
        return xDt

    def trig_poly_affine_dynamics(self, x: np.ndarray) -> Tuple:
        assert x.shape == (3,)
        if x.dtype == object:  # for verification
            f = np.array(
                [
                    sym.Polynomial((x[1] + 1) * x[2]),
                    sym.Polynomial(-x[0] * x[2]),
                    sym.Polynomial((self.g / self.l) * x[0]),
                ]
            )
            g = np.array(
                [
                    [sym.Polynomial()],
                    [sym.Polynomial()],
                    [sym.Polynomial(1 / (self.m * self.l**2))],
                ]
            )
        return (f, g)

    def trig_poly_state_eq_const(self, x: np.ndarray) -> sym.Polynomial:
        assert x.shape == (3,)
        return sym.Polynomial(x[0] ** 2 + (x[1] + 1) ** 2 - 1)

    def trig_poly_linearized_dynamics(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is used for the linearization of the dynamics.
        The equilibrium point is x_eq = [0, 0, 0] and u_eq = 0.
        """
        (f, g) = self.trig_poly_affine_dynamics(x)
        u = sym.MakeVectorContinuousVariable(g.shape[1], "u")
        x_eq = np.zeros((3,))
        u_eq = np.zeros((1,))

        gu = np.dot(g, u)
        Fxu = f + gu
        substitution = {x[i]: x_eq[i] for i in range(len(x_eq))}
        substitution.update({u[i]: u_eq[i] for i in range(len(u_eq))})

        A_symb = sym.Jacobian(Fxu, x)
        A = sym.Evaluate(A_symb, substitution)
        B = sym.Evaluate(g, substitution)

        return (A, B)
