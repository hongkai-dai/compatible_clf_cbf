from typing import Tuple
import numpy as np
import pydrake.symbolic as sym
import pydrake.systems.framework

"""
This is a simple adaptive cruise control example. In this example,
a car is controlled to follow a preceding car with a constant velocity v_ref.
The mission of the follower is to:
1. The desired following distance should be d_ref, the follower should keep a distance
    between [d_min, d_max]
2. keep tracking the velocity of the preceding car

variable definitions:
v_ref: the velocity of the preceding car
d_ref: the desired distance between the follower and the preceding car
d_min: the minimum distance between the follower and the preceding car
d_max: the maximum distance between the follower and the preceding car
v: the velocity of the follower
d: the distance between the follower and the preceding car
F: the control input of the follower

define the states:
x0: v_ref - v
x1: d - d_ref
u: acceleration of the follower

The system dynamics are defined as:
x0_dot = -u
x1_dot = x0
"""


class AdaptiveCruiseControlPlant(pydrake.systems.framework.LeafSystem):
    """
    This class defines the adaptive cruise control system. It is inherited
    from LeafSystem, meaning that we will use it for both the CLF-CBF
    verfication and synthesis, and also the simulation of CLF-CBF-QP
    as well.
    The initialization function defines the system parameters and
    also the input, state and output ports.
    The function DoCalcTimeDerivatives and trig_poly_dynamics
    defines the dynamics of the system. These functions are used for
    simulation.
    For verification and synthesis, we use the function
    affine_dynamics.
    """

    def __init__(self):
        super().__init__()
        self.m = 1500  # kg
        self.d_min = 1.0  # meter
        self.d_max = 5.0  # meter
        self.d_ref = 3.0  # meter
        self.v_ref = (20 * 1000) / 3600  # 20km/h, the unit of v_ref is m/s
        self.DeclareVectorInputPort(name="u", size=1)
        state_index = self.DeclareContinuousState(num_state_variables=2)
        self.DeclareStateOutputPort(name="x", state_index=state_index)

    def DoCalcTimeDerivatives(
        self,
        context: pydrake.systems.framework.Context,
        derivatives: pydrake.systems.framework.ContinuousState,
    ):
        x = context.get_continuous_state_vector().CopyToVector()
        u = self.EvalVectorInput(context, 0).CopyToVector()
        xdot: np.ndarray = self.system_dynamics(x, u)
        derivatives.SetFromVector(xdot)

    def system_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        The system dynamics of the adaptive cruise control system.
        """
        assert x.shape == (2,)
        assert u.shape == (1,)
        return np.array([-u[0], x[0]])

    def affine_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The affine dynamics of the adaptive cruise control system.
        """
        assert x.shape == (2,)
        f = np.array([sym.Polynomial(0), sym.Polynomial(x[0])])
        g = np.array([
            [sym.Polynomial(-1)],
            [sym.Polynomial(0)]
            ])
        return (f, g)

    def linear_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is used for the linearization of the dynamics.
        The equilibrium point is x_eq = [0, 0] and u_eq = 0.
        """
        (f, g) = self.affine_dynamics(x)
        u = sym.MakeVectorContinuousVariable(g.shape[1], "u")
        x_eq = np.zeros((2,))
        u_eq = np.zeros((1,))

        gu = np.dot(g, u)
        Fxu = f + gu
        substitution = {x[i]: x_eq[i] for i in range(len(x_eq))}
        substitution.update({u[i]: u_eq[i] for i in range(len(u_eq))})

        A_symb = sym.Jacobian(Fxu, x)
        A = sym.Evaluate(A_symb, substitution)
        B = sym.Evaluate(g, substitution)

        return (A, B)
