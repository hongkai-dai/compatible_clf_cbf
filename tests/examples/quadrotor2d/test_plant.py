import examples.quadrotor2d.plant as mut

import numpy as np
import jax.numpy as jnp
import jax
import pytest  # noqa
import pydrake.symbolic as sym


class TestQuadrotor2dPlant:
    def test_affine_dynamics(self):
        dut = mut.Quadrotor2dPlant()
        x = sym.MakeVectorContinuousVariable(6, "x")
        f, g = dut.affine_dynamics(x)
        assert f.shape == (6,)
        assert g.shape == (6, 2)

        def check(x_val, u_val):
            env = {x[i]: x_val[i] for i in range(6)}
            f_val = np.array([f[i].Evaluate(env) for i in range(6)])
            g_val = np.array(
                [[g[i, j].Evaluate(env) for j in range(2)] for i in range(6)]
            )
            xdot = dut.dynamics(x_val, u_val)
            xdot_expected = f_val + g_val @ u_val
            np.testing.assert_allclose(xdot, xdot_expected, atol=1e-6)

        check(np.array([0.5, 2, -0.3, 2, 4, 5]), np.array([0.3, 1.2]))

    def test_taylor_affine_dynamics(self):
        dut = mut.Quadrotor2dPlant()
        x = sym.MakeVectorContinuousVariable(6, "x")
        f, g = dut.affine_dynamics(x)
        f_taylor, g_taylor = dut.taylor_affine_dynamics(x)
        env = {x[i]: 0 for i in range(6)}

        f_taylor_expected = np.array(
            [sym.TaylorExpand(f[i], env, order=3) for i in range(6)]
        )
        g_taylor_expected = np.array(
            [
                [sym.TaylorExpand(g[i, j], env, order=3) for j in range(2)]
                for i in range(6)
            ]
        )

        def check(x_val):
            env_eval = {x[i]: x_val[i] for i in range(6)}
            f_taylor_val = np.array([f_taylor[i].Evaluate(env_eval) for i in range(6)])
            g_taylor_val = np.array(
                [
                    [g_taylor[i, j].Evaluate(env_eval) for j in range(2)]
                    for i in range(6)
                ]
            )
            f_taylor_expected_val = np.array(
                [f_taylor_expected[i].Evaluate(env_eval) for i in range(6)]
            )
            g_taylor_expected_val = np.array(
                [
                    [g_taylor_expected[i, j].Evaluate(env_eval) for j in range(2)]
                    for i in range(6)
                ]
            )
            np.testing.assert_allclose(f_taylor_val, f_taylor_expected_val, atol=1e-7)
            np.testing.assert_allclose(g_taylor_val, g_taylor_expected_val, atol=1e-7)

        check(np.array([0.5, 2, 0.3, -1.2, 0.4, 0.5]))
        check(np.array([1.5, 2, 1.3, -2.2, -0.9, 0.5]))

    def test_linearize_dynamics(self):
        dut = mut.Quadrotor2dPlant()
        x_des = np.array([0.5, 2, 1.2, 0.4, 0.5, 1.6])
        u_des = np.array([0.5, 1.7])
        A, B = dut.linearize_dynamics(x_des, u_des)
        x_jnp = jnp.array(x_des)
        u_jnp = jnp.array(u_des)
        AB_expected = jax.jacrev(lambda xu: dut.dynamics(xu[:6], xu[-2:]))(
            jnp.concatenate((x_jnp, u_jnp))
        )
        A_expected = AB_expected[:, :6]
        B_expected = AB_expected[:, 6:]
        np.testing.assert_allclose(A, A_expected, atol=1e-6)
        np.testing.assert_allclose(B, B_expected, atol=1e-6)


class TestQuadrotor2dTrigPlant:
    def test_dynamics(self):
        dut = mut.Quadrotor2dTrigPlant()
        x_non_trig = np.array([0.5, 0.3, 0.2, 2, 3, -1])
        x = dut.to_trig_state(x_non_trig)
        u = np.array([1, 3.0])
        xdot = dut.dynamics(x, u)
        assert xdot.shape == (7,)
        np.testing.assert_almost_equal((x[2:4] + np.array([0, 1])).dot(xdot[2:4]), 0, 6)

        quadrotor = mut.Quadrotor2dPlant()
        xdot_non_trig = quadrotor.dynamics(x_non_trig, u)
        np.testing.assert_almost_equal(x[:2], x_non_trig[:2])
        np.testing.assert_almost_equal(
            np.cos(x_non_trig[2]) * xdot_non_trig[2], xdot[2]
        )
        np.testing.assert_almost_equal(
            -np.sin(x_non_trig[2]) * xdot_non_trig[2], xdot[3]
        )
        np.testing.assert_almost_equal(xdot[-3:], xdot_non_trig[-3:])

    def test_affine_dynamics(self):
        dut = mut.Quadrotor2dTrigPlant()
        x = sym.MakeVectorContinuousVariable(7, "x")
        f, g = dut.affine_dynamics(x)
        assert f.shape == (7,)
        assert g.shape == (7, 2)

        def check(x_val, u_val):
            env = {x[i]: x_val[i] for i in range(7)}
            f_val = np.array([f[i].Evaluate(env) for i in range(7)])
            g_val = np.array(
                [[g[i, j].Evaluate(env) for j in range(2)] for i in range(7)]
            )
            xdot = dut.dynamics(x_val, u_val)
            xdot_expected = f_val + g_val @ u_val
            np.testing.assert_allclose(xdot, xdot_expected, atol=1e-5)

        check(np.array([1, 2, 0.5, np.sqrt(0.75) - 1, 0.3, 1.5, 2]), np.array([4, 2]))
        check(np.array([1, 2, -0.5, np.sqrt(0.75) - 1, 0.3, 1.5, 2]), np.array([4, 2]))

    def test_linearize_dynamics(self):
        dut = mut.Quadrotor2dTrigPlant()

        def check(x_val, u_val):
            A, B = dut.linearize_dynamics(x_val, u_val)

            x_jnp = jnp.array(x_val)
            u_jnp = jnp.array(u_val)
            xu_jnp = jnp.concatenate((x_jnp, u_jnp))

            AB = jax.jacrev(lambda xu: dut.dynamics(xu[:7], xu[7:]))(xu_jnp)
            A_expected = np.array(AB[:, :7])
            B_expected = np.array(AB[:, 7:])
            np.testing.assert_allclose(A, A_expected, atol=1e-6)
            np.testing.assert_allclose(B, B_expected, atol=1e-6)

        check(np.array([1, 2, 0.5, np.sqrt(0.75) - 1, 3, 2, 4]), np.array([2, 3]))
        check(np.array([4, 2, -0.5, np.sqrt(0.75) - 1, 3, -2, 4]), np.array([4, 3]))

    def test_equality_constraint(self):
        dut = mut.Quadrotor2dTrigPlant()
        x = sym.MakeVectorContinuousVariable(7, "x")
        equality_constraint = dut.equality_constraint(x)
        assert equality_constraint.shape == (1,)

        def check(x_non_trig):
            x_trig = dut.to_trig_state(x_non_trig)
            env = {x[i]: x_trig[i] for i in range(7)}
            np.testing.assert_allclose(
                equality_constraint[0].Evaluate(env), 0, atol=1e-5
            )

        check(np.array([2, 3, 0.5, 3, 4, 5]))
        check(np.array([2, 3, -1.5, 4, 4, 5]))
