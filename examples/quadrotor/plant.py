from typing import Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import jax
import jax.numpy as jnp
import pydrake.symbolic as sym
import pydrake.examples
import pydrake.systems.framework
import pydrake.geometry
import pydrake.multibody.plant
import pydrake.multibody.parsing
import pydrake.math
from pydrake.common.value import Value


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


def calc_matrix_relating_rpydt_to_angular_velocity_in_child(
    rpy: np.ndarray, T
) -> np.ndarray:
    """
    This is the same as CalcMatrixRelatingRpyDtToAngularVelocityInChild in Drake,
    but supports symbolic computation.
    """
    sp = np.sin(rpy[1])
    cp = np.cos(rpy[1])
    one_over_cp = 1.0 / cp
    sr = np.sin(rpy[0])
    cr = np.cos(rpy[0])
    cr_over_cp = cr * one_over_cp
    sr_over_cp = sr * one_over_cp
    M = np.array(
        [
            [T(1), sr_over_cp * sp, cr_over_cp * sp],
            [T(0), cr, -sr],
            [T(0), sr_over_cp, cr_over_cp],
        ]
    )
    return M


class QuadrotorPlant(pydrake.systems.framework.LeafSystem):
    """
    The state is (pos, rpy, pos_dot, omega_WB_B)
    """

    m: float
    I: np.ndarray
    l: float
    g: float
    kF: float
    kM: float

    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort("thrust", 4)
        state_index = self.DeclareContinuousState(12)
        self.DeclareStateOutputPort("x", state_index)
        drake_plant = pydrake.examples.QuadrotorPlant()
        self.m = drake_plant.m()
        self.I = drake_plant.inertia()
        self.g = drake_plant.g()
        self.l = drake_plant.length()
        self.kF = drake_plant.force_constant()
        self.kM = drake_plant.moment_constant()
        self.I_inv = np.linalg.inv(self.I)

    def DoCalcTimeDerivatives(
        self,
        context: pydrake.systems.framework.Context,
        derivatives: pydrake.systems.framework.ContinuousState,
    ):
        x = context.get_continuous_state_vector().CopyToVector()
        u = self.EvalVectorInput(context, 0).CopyToVector()
        xdot: np.ndarray = self.dynamics(x, u)
        derivatives.SetFromVector(xdot)

    def dynamics(self, x: np.ndarray, u: np.ndarray, T=float) -> np.ndarray:
        uF_Bz = self.kF * u

        Faero_B = uF_Bz.sum() * np.array([0, 0, 1])
        Mx = self.l * (uF_Bz[1] - uF_Bz[3])
        My = self.l * (uF_Bz[2] - uF_Bz[0])
        uTau_Bz = self.kM * u
        Mz = uTau_Bz[0] - uTau_Bz[1] + uTau_Bz[2] - uTau_Bz[3]

        tau_B = np.stack((Mx, My, Mz))
        Fgravity_N = np.array([0, 0, -self.m * self.g])

        rpy = pydrake.math.RollPitchYaw_[T](x[3:6])
        w_NB_B = x[-3:]
        rpy_dot = rpy.CalcRpyDtFromAngularVelocityInChild(w_NB_B)
        R_NB = pydrake.math.RotationMatrix_[T](rpy)

        xyzDDt = (Fgravity_N + R_NB @ Faero_B) / self.m

        wIw = np.cross(w_NB_B, self.I @ w_NB_B)
        alpha_NB_B = self.I_inv @ (tau_B - wIw)

        xDt = np.concatenate((x[6:9], rpy_dot, xyzDDt, alpha_NB_B))

        return xDt

    def affine_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = sym.Expression if x.dtype == object else float
        f = np.zeros((12,), dtype=x.dtype)
        g = np.zeros((12, 4), dtype=x.dtype)
        for i in range(12):
            f[i] = T(0)
            for j in range(4):
                g[i, j] = T(0)

        f[:3] = np.array([T(x[i]) for i in range(6, 9)])
        f[3:6] = (
            calc_matrix_relating_rpydt_to_angular_velocity_in_child(x[3:6], T=T)
            @ x[-3:]
        )
        rpy = pydrake.math.RollPitchYaw_[T](x[3], x[4], x[5])
        R = pydrake.math.RotationMatrix_[T](rpy)
        f[8] = T(-self.g)
        g[6:9, :] = np.outer(R.matrix()[:, 2], self.kF * np.ones((4,))) / self.m
        w_NB_B = x[-3:]
        f[9:] = -self.I_inv @ np.cross(w_NB_B, self.I @ w_NB_B)
        # tau_B = u_to_M * u
        u_to_M = np.array(
            [
                [T(0), T(self.l * self.kF), T(0), T(-self.l * self.kF)],
                [T(-self.l * self.kF), T(0), T(self.l * self.kF), T(0)],
                [T(self.kM), T(-self.kM), T(self.kM), T(-self.kM)],
            ]
        )
        g[9:, :] = self.I_inv @ u_to_M
        return f, g

    def affine_dynamics_taylor(
        self, x: np.ndarray, x_val: np.ndarray, f_degree: int, g_degree: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Taylor-expand the control-affine dynamics xdot = f(x) + g(x) * u.
        Return f(x) and g(x) in the Taylor expansion.
        """
        env = {x[i]: x_val[i] for i in range(12)}
        f_expr, g_expr = self.affine_dynamics(x)
        f = np.array(
            [
                sym.Polynomial(sym.TaylorExpand(f_expr[i], env, order=f_degree))
                for i in range(12)
            ]
        )
        g = np.array(
            [
                [
                    sym.Polynomial(sym.TaylorExpand(g_expr[i, j], env, order=g_degree))
                    for j in range(4)
                ]
                for i in range(12)
            ]
        )
        return (f, g)


class QuadrotorPolyPlant(pydrake.systems.framework.LeafSystem):
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
        super().__init__()
        self.DeclareVectorInputPort("thrust", 4)
        state_index = self.DeclareContinuousState(13)
        self.DeclareStateOutputPort("x", state_index)
        drake_plant = pydrake.examples.QuadrotorPlant()
        self.m = drake_plant.m()
        self.I = drake_plant.inertia()
        self.g = drake_plant.g()
        self.l = drake_plant.length()
        self.kF = drake_plant.force_constant()
        self.kM = drake_plant.moment_constant()
        self.I_inv = np.linalg.inv(self.I)

    def DoCalcTimeDerivatives(
        self,
        context: pydrake.systems.framework.Context,
        derivatives: pydrake.systems.framework.ContinuousState,
    ):
        x = context.get_continuous_state_vector().CopyToVector()
        u = self.EvalVectorInput(context, 0).CopyToVector()
        xdot: np.ndarray = self.dynamics(x, u)
        derivatives.SetFromVector(xdot)

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
        if x.dtype == object:
            data_type = sym.Polynomial
        else:
            data_type = float
        assert x.shape == (13,)
        f = np.empty((13,), dtype=x.dtype)
        g = np.empty((13, 4), dtype=x.dtype)
        quat = x[:4] + np.array([1, 0, 0, 0])
        w_NB_B = x[-3:]
        f[0] = data_type(
            0.5 * (-w_NB_B[0] * quat[1] - w_NB_B[1] * quat[2] - w_NB_B[2] * quat[3])
        )
        f[1] = data_type(
            0.5 * (w_NB_B[0] * quat[0] + w_NB_B[2] * quat[2] - w_NB_B[1] * quat[3])
        )
        f[2] = data_type(
            0.5 * (w_NB_B[1] * quat[0] - w_NB_B[2] * quat[1] + w_NB_B[0] * quat[3])
        )
        f[3] = data_type(
            0.5 * (w_NB_B[2] * quat[0] + w_NB_B[1] * quat[1] - w_NB_B[0] * quat[2])
        )
        f[4] = data_type(x[7])
        f[5] = data_type(x[8])
        f[6] = data_type(x[9])
        for i in range(7):
            for j in range(4):
                g[i][j] = data_type(0)

        f[7] = data_type(0)
        f[8] = data_type(0)
        f[9] = data_type(-self.g)

        R_NB = quat2rotmat(quat)

        for i in range(3):
            for j in range(4):
                g[i + 7][j] = data_type(R_NB[i, 2] * self.kF / self.m)

        wIw = np.cross(w_NB_B, self.I @ w_NB_B)
        for i in range(3):
            f[10 + i] = data_type(-wIw[i] / self.I[i, i])
        g[10, 0] = data_type(0)
        g[10, 1] = data_type(self.kF * self.l / self.I[0, 0])
        g[10, 2] = data_type(0)
        g[10, 3] = -g[10, 1]
        g[11, 0] = data_type(-self.kF * self.l / self.I[1, 1])
        g[11, 1] = data_type(0)
        g[11, 2] = -g[11, 0]
        g[11, 3] = data_type(0)
        g[12, 0] = data_type(self.kF * self.kM / self.I[2, 2])
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


class QuadrotorPolyGeometry(pydrake.systems.framework.LeafSystem):
    def __init__(
        self, scene_graph: pydrake.geometry.SceneGraph, name: Optional[str] = None
    ):
        super().__init__()
        mbp = pydrake.multibody.plant.MultibodyPlant(0.0)
        parser = pydrake.multibody.parsing.Parser(
            mbp, scene_graph, "quadrotor" if name is None else name
        )
        model_instance_indices = parser.AddModelsFromUrl(
            "package://drake_models/skydio_2/quadrotor.urdf"
        )
        mbp.Finalize()

        body_indices = mbp.GetBodyIndices(model_instance_indices[0])
        body_index = body_indices[0]
        self.source_id = mbp.get_source_id()
        self.frame_id = mbp.GetBodyFrameIdOrThrow(body_index)

        self.DeclareVectorInputPort("state", 13)
        self.DeclareAbstractOutputPort(
            "geometry_pose",
            lambda: Value[pydrake.geometry.FramePoseVector](),
            self.output_geometry_pose,
        )

    def output_geometry_pose(
        self,
        context: pydrake.systems.framework.Context,
        poses,
    ):
        state = self.get_input_port(0).Eval(context)

        pose = pydrake.math.RigidTransform(
            pydrake.common.eigen_geometry.Quaternion(
                state[0] + 1, state[1], state[2], state[3]
            ),
            state[4:7],
        )
        poses_value = pydrake.geometry.FramePoseVector()
        poses_value.set_value(self.frame_id, pose)
        poses.set_value(poses_value)

    @staticmethod
    def AddToBuilder(
        builder: pydrake.systems.framework.DiagramBuilder,
        quadrotor_state_port: pydrake.systems.framework.OutputPort,
        name: str,
        scene_graph: pydrake.geometry.SceneGraph,
    ) -> Self:
        quadrotor_geometry = builder.AddSystem(QuadrotorPolyGeometry(scene_graph, name))
        builder.Connect(quadrotor_state_port, quadrotor_geometry.get_input_port(0))
        builder.Connect(
            quadrotor_geometry.get_output_port(0),
            scene_graph.get_source_pose_port(quadrotor_geometry.source_id),
        )
        return quadrotor_geometry
