from typing import Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
import pydrake.symbolic as sym


def quat2rotmat(quat: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(quat, np.ndarray):
        R = np.empty((3, 3), dtype=quat.dtype)
        numpy_type = np
    elif isinstance(quat, jnp.ndarray):
        R = jnp.empty((3, 3))
        numpy_type = jnp

    def set_val(i, j, val):
        if isinstance(R, np.ndarray):
            R[i, j] = val
        else:
            R.at[i, j].set(val)

    R00 = 1 - 2 * quat[2] ** 2 - 2 * quat[3] ** 2
    R01 = 2 * quat[1] * quat[2] - 2 * quat[0] * quat[3]
    R02 = 2 * quat[1] * quat[3] + 2 * quat[0] * quat[2]
    R10 = 2 * quat[1] * quat[2] + 2 * quat[0] * quat[3]
    R11 = 1 - 2 * quat[1] ** 2 - 2 * quat[3] ** 2
    R12 = 2 * quat[2] * quat[3] - 2 * quat[0] * quat[1]
    R20 = 2 * quat[1] * quat[3] - 2 * quat[0] * quat[2]
    R21 = 2 * quat[2] * quat[3] + 2 * quat[0] * quat[1]
    R22 = 1 - 2 * quat[1] ** 2 - 2 * quat[2] ** 2
    R = numpy_type.stack((R00, R01, R02, R10, R11, R12, R20, R21, R22)).reshape((3, 3))
    return R


class QuadrotorPolyPlant:
    """
    The state is x=(quat - [1, 0, 0, 0], pos, pos_dot, omega_WB_B)
    Note that the equilibrium is at x=0.
    The dynamics is a polynomial function of the state/action.
    """

    m: float
    I: np.ndarray
    l: float
    g: float
    kF: float
    kM: float

    def __init__(self):
        self.m = 0.775
        self.I = np.array([[0.0015, 0, 0], [0, 0.0025, 0], [0, 0, 0.0035]])
        self.g = 9.81
        self.l = 0.15
        self.kF = 1.0
        self.kM = 0.0245
        self.I_inv = np.linalg.inv(self.I)

    def dynamics(
        self, x: Union[np.ndarray, jnp.ndarray], u: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        """
        Dynamics xdot = f(x, u)
        """
        if isinstance(x, np.ndarray):
            numpy_type = np
            np_array = np.array
        elif isinstance(x, jnp.ndarray):
            numpy_type = jnp
            np_array = jnp.array

        uF_Bz = self.kF * u

        Faero_B = uF_Bz.sum() * np_array([0, 0, 1])
        Mx = self.l * (uF_Bz[1] - uF_Bz[3])
        My = self.l * (uF_Bz[2] - uF_Bz[0])
        uTau_Bz = self.kM * u
        Mz = uTau_Bz[0] - uTau_Bz[1] + uTau_Bz[2] - uTau_Bz[3]

        tau_B = numpy_type.stack((Mx, My, Mz))
        Fgravity_N = np_array([0, 0, -self.m * self.g])

        quat = x[:4] + np_array([1, 0, 0, 0])
        w_NB_B = x[-3:]
        quat_dot = numpy_type.stack(
            [
                0.5
                * (-w_NB_B[0] * quat[1] - w_NB_B[1] * quat[2] - w_NB_B[2] * quat[3]),
                0.5 * (w_NB_B[0] * quat[0] + w_NB_B[2] * quat[2] - w_NB_B[1] * quat[3]),
                0.5 * (w_NB_B[1] * quat[0] - w_NB_B[2] * quat[1] + w_NB_B[0] * quat[3]),
                0.5 * (w_NB_B[2] * quat[0] + w_NB_B[1] * quat[1] - w_NB_B[0] * quat[2]),
            ]
        )
        R_NB = quat2rotmat(quat)

        xyzDDt = (Fgravity_N + R_NB @ Faero_B) / self.m

        wIw = numpy_type.cross(w_NB_B, self.I @ w_NB_B)
        alpha_NB_B = self.I_inv @ (tau_B - wIw)

        xDt = numpy_type.concatenate((quat_dot, x[7:10], xyzDDt, alpha_NB_B))

        return xDt

    def affine_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the state x (as a vector of symbolic variables), return the affine
        dynamics xdot = f(x) + g(x) * u, where f(x) is a vector of symbolic
        polynomials, and g(x) is a matrix of symbolic polynomials.
        """
        assert x.shape == (13,)
        f = np.empty((13,), dtype=object)
        g = np.empty((13, 4), dtype=object)
        quat = x[:4] + np.array([1, 0, 0, 0])
        w_NB_B = x[-3:]
        f[0] = sym.Polynomial(
            0.5 * (-w_NB_B[0] * quat[1] - w_NB_B[1] * quat[2] - w_NB_B[2] * quat[3])
        )
        f[1] = sym.Polynomial(
            0.5 * (w_NB_B[0] * quat[0] + w_NB_B[2] * quat[2] - w_NB_B[1] * quat[3])
        )
        f[2] = sym.Polynomial(
            0.5 * (w_NB_B[1] * quat[0] - w_NB_B[2] * quat[1] + w_NB_B[0] * quat[3])
        )
        f[3] = sym.Polynomial(
            0.5 * (w_NB_B[2] * quat[0] + w_NB_B[1] * quat[1] - w_NB_B[0] * quat[2])
        )
        f[4] = sym.Polynomial(x[7])
        f[5] = sym.Polynomial(x[8])
        f[6] = sym.Polynomial(x[9])
        for i in range(7):
            for j in range(4):
                g[i][j] = sym.Polynomial()

        f[7] = sym.Polynomial()
        f[8] = sym.Polynomial()
        f[9] = sym.Polynomial(-self.g)

        R_NB = quat2rotmat(quat)

        for i in range(3):
            for j in range(4):
                g[i + 7][j] = sym.Polynomial(R_NB[i, 2] * self.kF / self.m)

        wIw = np.cross(w_NB_B, self.I @ w_NB_B)
        for i in range(3):
            f[10 + i] = sym.Polynomial(-wIw[i] / self.I[i, i])
        g[10, 0] = sym.Polynomial()
        g[10, 1] = sym.Polynomial(self.kF * self.l / self.I[0, 0])
        g[10, 2] = sym.Polynomial()
        g[10, 3] = -g[10, 1]
        g[11, 0] = sym.Polynomial(-self.kF * self.l / self.I[1, 1])
        g[11, 1] = sym.Polynomial()
        g[11, 2] = -g[11, 0]
        g[11, 3] = sym.Polynomial()
        g[12, 0] = sym.Polynomial(self.kF * self.kM / self.I[2, 2])
        g[12, 1] = -g[12, 0]
        g[12, 2] = -g[12, 1]
        g[12, 3] = -g[12, 0]
        return (f, g)

    def equality_constraint(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the left hand side of the equality constraint
        quaternion.squared_norm() - 1 = 0
        """
        return np.array(
            [sym.Polynomial(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + 2 * x[0])]
        )

    def linearize_dynamics(
        self, x_val: np.ndarray, u_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For the dynamics function xdot = f(x, u), compute the linearization as
        A = ∂f/∂x, B = ∂f/∂u
        """

        x_jnp = jnp.array(x_val)
        u_jnp = jnp.array(u_val)

        def f(xu):
            return self.dynamics(xu[:13], xu[-4:])

        xu_jnp = jnp.concatenate((x_jnp, u_jnp))
        AB_jnp = jax.jacfwd(f)(xu_jnp)
        AB = np.array(AB_jnp)
        return AB[:, :13], AB[:, 13:]
