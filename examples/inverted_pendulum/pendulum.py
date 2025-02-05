from typing import Tuple
import numpy as np
import pydrake.symbolic as sym

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

def affine_dynamics(x: np.ndarray) -> Tuple:
    """
    This should be used of the simulation of the inverted pendulum.
    """
    gravity = 9.81
    length = 0.5
    mass = 2
    assert x.shape == (2,)
    f = np.array([
        x[1],
        (gravity/length)*np.sin(x[0])
        ])
    g = np.array([
        [0],
        [1/(mass*length**2)]
        ])
    return (f, g)

def affine_trig_poly_dynamics(x: np.ndarray) -> Tuple:
    gravity = 9.81
    length = 0.5
    mass = 2
    assert x.shape == (3,)
    if x.dtype==object:
        f = np.array([
            sym.Polynomial((x[1]+1)*x[2]),
            sym.Polynomial(-x[0]*x[2]),
            sym.Polynomial((gravity/length)*x[0])
        ])
        g = np.array([[sym.Polynomial()],
                      [sym.Polynomial()],
                      [sym.Polynomial(1/(mass*length**2))]
                      ])
        
    else:
        f = np.array([
            (x[1]+1)*x[2], 
            -x[0]*x[2],
            (gravity/length)*x[0]
            ])
        g = np.array([
            [0],
            [0],
            [1/(mass*length**2)]
            ])
    
    return (f, g)

def affine_trig_poly_state_constraints(x: np.ndarray) -> sym.Polynomial:
    return sym.Polynomial(x[0]**2 + x[1]**2 + 2*x[1])

def affine_approximate_linear_dynamics(x: np.ndarray) -> Tuple:
    """
    This dynmaics can be used to verify the compatibility when the 
    safe region of the \theta is very small, say, \theta \in [-pi/18, pi/18]
    , which means the angle \theta should be kept with in [-10, 10] degrees.
    In this case, we can approximate the sin(\theta) by \theta. 
    """
    gravity = 9.81
    length = 1
    mass = 5
    assert x.shape == (2,)
    if x.dtype==object:
        f = np.array([
            sym.Polynomial(x[1]),
            sym.Polynomial((gravity/length)*x[0])
        ])
        g = np.array([[sym.Polynomial()],
                      [sym.Polynomial(1/(mass*length**2))]
                      ])
        
    else:
        f = np.array([
            x[1],
            (gravity/length)*x[0]
            ])
        g = np.array([
            [0],
            [1/(mass*length**2)]
            ])
    
    return (f, g)

def affine_approximate_nonlinear_dynamics(x: np.ndarray) -> Tuple:
    """
    This dynmaics can be used to verify the compatibility when the 
    safe region of the \theta is relatively larger, say, \theta \in [-pi/4, pi/4]
    , which means the angle \theta should be kept with in [-45, 45] degrees.
    In this case, we can approximate the sin(\theta) by \theta - \theta^3/6. 
    """
    gravity = 9.81
    length = 1
    mass = 5
    assert x.shape == (2,)
    if x.dtype==object:
        f = np.array([
            sym.Polynomial(x[1]),
            sym.Polynomial((gravity/length)*(x[0] - (x[0]**3)/6))
        ])
        g = np.array([[sym.Polynomial()],
                      [sym.Polynomial(1/(mass*length**2))]
                      ])
        
    else:
        f = np.array([
            x[1],
            (gravity/length)*(x[0] - (x[0]**3)/6)
            ])
        g = np.array([
            [0],
            [1/(mass*length**2)]
            ])
    
    return (f, g)