import examples.quadrotor.plant as mut

import numpy as np
import pytest  # noqa
import pydrake.symbolic as sym
import pydrake.math


def test_quat2rotmat():
    quat = np.array([1, 2, 3, 4.0])
    quat = quat / np.linalg.norm(quat)
    R = mut.quat2rotmat(quat)
    R_expected = pydrake.math.RotationMatrix(
        pydrake.common.eigen_geometry.Quaternion(quat)
    ).matrix()
    np.testing.assert_allclose(R, R_expected)


class TestQuadrotorPolyPlant:
    def test_dynamics(self):
        dut = mut.QuadrotorPolyPlant()
        quat = np.array([0.3, -0.4, 0.5, -np.sqrt(2) / 2])
        pos = np.array([1, 2, 3.0])
        pos_dot = np.array([0.5, -0.3, 0.2])
        omega_WB_B = np.array([0.3, 0.4, 0.5])
        u = np.array([1, 2, 3, 4.0])
        x = np.concatenate((quat - np.array([1, 0, 0, 0]), pos, pos_dot, omega_WB_B))
        xdot = dut.dynamics(x, u)
        assert xdot.shape == (13,)
        quat_dot = xdot[:4]
        np.testing.assert_almost_equal(quat.dot(quat_dot), 0, 6)

    def test_affine_dynamics(self):
        dut = mut.QuadrotorPolyPlant()
        x = sym.MakeVectorContinuousVariable(13, "x")
        f, g = dut.affine_dynamics(x)
        assert f.shape == (13,)
        assert g.shape == (13, 4)

        def compare_xdot(quat_val, pos_val, pos_dot_val, omega_val, u_val):
            x_val = np.concatenate(
                (quat_val - np.array([1, 0, 0, 0.0]), pos_val, pos_dot_val, omega_val)
            )
            xdot = dut.dynamics(x_val, u_val)
            env = {x[i]: x_val[i] for i in range(13)}
            f_val = np.array([f[i].Evaluate(env) for i in range(13)])
            g_val = np.array(
                [[g[i, j].Evaluate(env) for j in range(4)] for i in range(13)]
            )
            xdot_expected = f_val + g_val @ u_val
            np.testing.assert_allclose(xdot, xdot_expected)

        compare_xdot(
            np.array([1, 0, 0, 0.0]),
            np.array([1, 2, 3.0]),
            np.array([0.5, -1, 2]),
            np.array([2, 3, 1]),
            np.array([1, 2, 3, 4]),
        )
        compare_xdot(
            np.array([0.3, 0.4, 0.5, np.sqrt(2) / 2]),
            np.array([1, 2, 3.0]),
            np.array([0.5, -1, 2]),
            np.array([2, 3, 1]),
            np.array([1, 2, 3, 4]),
        )

    def test_equality_constraint(self):
        dut = mut.QuadrotorPolyPlant()
        x = sym.MakeVectorContinuousVariable(13, "x")
        state_eq_constraints = dut.equality_constraint(x)
        assert state_eq_constraints.shape == (1,)

        def check_constraint(quat_val, pos_val, pos_dot_val, omega_val):
            x_val = np.concatenate(
                (quat_val - np.array([1, 0, 0, 0]), pos_val, pos_dot_val, omega_val)
            )
            env = {x[i]: x_val[i] for i in range(13)}
            val = state_eq_constraints[0].Evaluate(env)
            np.testing.assert_allclose(val, 0, atol=1e-5)

        check_constraint(
            np.array([0.3, 0.4, 0.5, np.sqrt(2) / 2]),
            np.array([1, 2, 3.0]),
            np.array([0.5, -1, 2]),
            np.array([2, 3, 1]),
        )

    def test_linearize_dynamics(self):
        dut = mut.QuadrotorPolyPlant()
        x = sym.MakeVectorContinuousVariable(13, "x")
        f, g = dut.affine_dynamics(x)

        def check(x_val, u_val):
            A, B = dut.linearize_dynamics(x_val, u_val)
            env = {x[i]: x_val[i] for i in range(13)}
            g_val = np.array(
                [[g[i, j].Evaluate(env) for j in range(4)] for i in range(13)]
            )
            np.testing.assert_allclose(B, g_val, atol=1e-5)

        x_val = np.array(
            [-0.7, 0.4, 0.5, np.sqrt(2) / 2, 0.2, 1, 4, -2, 3, 1, 0.5, 1, 2]
        )
        u_val = np.array([1, 2, 3, 4.0])
        A, B = dut.linearize_dynamics(x_val, u_val)
        check(x_val, u_val)
