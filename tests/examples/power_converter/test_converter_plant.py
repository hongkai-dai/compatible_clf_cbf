import examples.power_converter.plant as mut

import numpy as np
import jax
import jax.numpy as jnp

import pydrake.symbolic as sym


def test_dynamics():
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = mut.affine_dynamics(x)

    def test(x_val, u_val):
        xdot_val = mut.dynamics(x_val, u_val)
        env = {x[i]: x_val[i] for i in range(3)}
        f_val = np.array([f_i.Evaluate(env) for f_i in f])
        g_val = np.array([[g[i, j].Evaluate(env) for j in range(2)] for i in range(3)])
        xdot_val_expected = f_val + g_val @ u_val
        np.testing.assert_allclose(xdot_val, xdot_val_expected, atol=1e-6)

    test(np.array([-3.2, 1.5, 3.9]), np.array([0.4, 1.7]))


def test_linearize_dynamics():
    x_des = np.array([0.5, 1.4, -2.3])
    u_des = np.array([4.5, -2.1])
    A, B = mut.linearize_dynamics(x_des, u_des)
    x_jnp = jnp.array(x_des)
    u_jnp = jnp.array(u_des)
    AB_expected = jax.jacrev(lambda xu: mut.dynamics(xu[:3], xu[3:]))(
        jnp.concatenate((x_jnp, u_jnp))
    )
    A_expected = AB_expected[:, :3]
    B_expected = AB_expected[:, 3:]
    np.testing.assert_allclose(A, A_expected, atol=1e-6)
    np.testing.assert_allclose(B, B_expected, atol=1e-6)
