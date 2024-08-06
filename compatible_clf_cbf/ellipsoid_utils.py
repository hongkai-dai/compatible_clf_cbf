"""
The utility function for manipulating ellipsoids
"""

from typing import List, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.lines
import numpy as np

import pydrake.solvers as solvers
import pydrake.symbolic as sym

import compatible_clf_cbf.utils as utils


def add_max_volume_linear_cost(
    prog: solvers.MathematicalProgram,
    S: np.ndarray,
    b: np.ndarray,
    c: sym.Variable,
    S_bar: np.ndarray,
    b_bar: np.ndarray,
    c_bar: float,
) -> solvers.Binding[solvers.LinearCost]:
    """
    Adds the linear cost as the linearization of an objective which correlates
    to the volume of the ellipsoid ℰ={x | xᵀSx+bᵀx+c≤0}

    To maximize the ellipsoid voluem, we can maximize
    log(bᵀS⁻¹b/4-c) - 1/n * log(det(S))

    The linearization of this function at (S̅, b̅, c̅) is
    max trace(⌈c   bᵀ/2⌉ᵀ  ⌈c̅,   b̅ᵀ/2⌉⁻¹) -(1+1/n)*trace(Sᵀ * S̅⁻¹)
              ⌊b/2    S⌋   ⌊b̅/2     S̅⌋

    Check doc/maximize_inner_ellipsoid.md for more details.

    Args:
      prog: The optimization problem to which the cost is added.
      S: A symmetric matrix of decision variables. S must have been registered
        in `prog` already.
      b: A vector of decision variables. b must have been registered in `prog`
        already.
      c: A decision variable. c must have been registered in `prog` already.
      S_bar: A symmetric matrix of floats, where we linearize the objective.
      b_bar: A vector of floats, where we linearized the objective.
    """

    n = S.shape[0]
    assert S.shape == (n, n)
    assert b.shape == (n,)
    assert S_bar.shape == (n, n)
    assert b_bar.shape == (n,)
    mat = np.empty((n + 1, n + 1), dtype=object)
    mat[0, 0] = c
    mat[0, 1:] = b / 2
    mat[1:, 0] = b / 2
    mat[1:, 1:] = S

    mat_bar = np.empty((n + 1, n + 1))
    mat_bar[0, 0] = c_bar
    mat_bar[0, 1:] = b_bar / 2
    mat_bar[1:, 0] = b_bar / 2
    mat_bar[1:, 1:] = S_bar
    mat_bar_inv = np.linalg.inv(mat_bar)
    S_bar_inv = np.linalg.inv(S_bar)

    cost_expr = np.trace(mat.T @ mat_bar_inv) - (1 + 1.0 / n) * np.trace(
        S.T @ S_bar_inv
    )
    cost = prog.AddLinearCost(-cost_expr)
    return cost


def _add_ellipsoid_trust_region(
    prog: solvers.MathematicalProgram,
    S: np.ndarray,
    b: np.ndarray,
    c: sym.Variable,
    S_bar: np.ndarray,
    b_bar: np.ndarray,
    c_bar: float,
    trust_region,
):
    """
    Add the constraint (S - S_bar)^2 + (b-b_bar)^2 + (c-c_bar)^2 <= trust_region.
    """
    dim = S.shape[0]
    linear_coeff = np.concatenate(
        (
            utils.to_lower_triangular_columns(S_bar).reshape((-1,)),
            b_bar.reshape((-1,)),
            np.array([c_bar]),
        )
    )
    constraint = prog.AddQuadraticAsRotatedLorentzConeConstraint(
        np.eye(int((dim + 1) * dim / 2) + dim + 1),
        -linear_coeff,
        linear_coeff.dot(linear_coeff) / 2 - trust_region,
        np.concatenate(
            (
                utils.to_lower_triangular_columns(S).reshape((-1,)),
                b.reshape((-1,)),
                np.array([c]),
            )
        ),
    )
    return constraint


def maximize_inner_ellipsoid_sequentially(
    prog: solvers.MathematicalProgram,
    S: np.ndarray,
    b: np.ndarray,
    c: sym.Variable,
    S_init: np.ndarray,
    b_init: np.ndarray,
    c_init: float,
    max_iter: int = 10,
    convergence_tol: float = 1e-3,
    solver_id: Optional[solvers.SolverId] = None,
    solver_options: Optional[solvers.SolverOptions] = None,
    trust_region: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Maximize the inner ellipsoid ℰ={x | xᵀSx+bᵀx+c≤0} through a sequence of
    convex programs.

    Args:
      prog: The optimization program that already contains all the constraints
        that the ellipsoid is inside the set.
      S: A symmetric matrix of decision variables. S must have been registered
        in `prog` already.
      b: A vector of decision variables. b must have been registered in `prog`
        already.
      c: A decision variable. c must have been registered in `prog` already.
      S_init: The initial guess of S.
      b_init: The initial guess of b.
      c_init: The initial guess of c.
    """
    S_bar = S_init
    b_bar = b_init
    c_bar = c_init

    def volume(S: np.ndarray, b: np.ndarray, c: float) -> float:
        n = S.shape[0]
        return (b.dot(np.linalg.solve(S, b)) / 4 - c) ** (n / 2) / np.sqrt(
            np.linalg.det(S)
        )

    cost = None
    volume_prev = volume(S_init, b_init, c_init)
    assert max_iter >= 1
    trust_region_constraint = None
    for i in range(max_iter):
        if cost is not None:
            prog.RemoveCost(cost)
        cost = add_max_volume_linear_cost(prog, S, b, c, S_bar, b_bar, c_bar)
        # The center of the ellipsoid {x | xᵀS_bar*x+b_barᵀx+c_bar≤0}
        ellipsoid_center = -np.linalg.solve(S_bar, b_bar) / 2
        # To constrain that {x | xᵀSx+bᵀx+c≤0} is a valid ellipsoid, we want it
        # to contain at least one point. We choose that contained point to be
        # ellipsoid_center.
        prog.AddLinearConstraint(
            ellipsoid_center.dot(S @ ellipsoid_center) + np.dot(b, ellipsoid_center) + c
            <= 0
        )

        if trust_region_constraint is not None:
            prog.RemoveConstraint(trust_region_constraint)
        if trust_region is not None:
            assert trust_region >= 0
            trust_region_constraint = _add_ellipsoid_trust_region(
                prog, S, b, c, S_bar, b_bar, c_bar, trust_region
            )
        result = utils.solve_with_id(prog, solver_id, solver_options)
        assert (
            result.is_success()
        ), f"Fails in iter={i} with {result.get_solution_result()}"
        S_result = result.GetSolution(S)
        b_result = result.GetSolution(b)
        c_result = result.GetSolution(c)
        volume_result = volume(S_result, b_result, c_result)
        if volume_result - volume_prev <= convergence_tol:
            break
        else:
            volume_prev = volume_result
            S_bar = S_result
            b_bar = b_result
            c_bar = c_result
    return S_result, b_result, c_result


def add_minimize_ellipsoid_volume(
    prog: solvers.MathematicalProgram, S: np.ndarray, b: np.ndarray, c: sym.Variable
) -> sym.Variable:
    """
    Minimize the volume of the ellipsoid {x | xᵀSx + bᵀx + c ≤ 0}
    where S, b, and c are decision variables.

    See doc/minimize_ellipsoid_volume.md for the details (you will need to
    enable MathJax in your markdown viewer).

    We minimize the volume through the convex program
    min r
    s.t ⌈c+r  bᵀ/2⌉ is psd
        ⌊b/2     S⌋

        log det(S) >= 0

    Args:
      S: a symmetric matrix of decision variables. S must have been registered
      in `prog` already. It is the user's responsibility to impose "S is psd".
      b: a vector of decision variables. b must have been registered in `prog` already.
      c: a symbolic Variable. c must have been registered in `prog` already.
    Returns:
      r: The slack decision variable.
    """
    x_dim = S.shape[0]
    assert S.shape == (x_dim, x_dim)
    r = prog.NewContinuousVariables(1, "r")[0]
    prog.AddLinearCost(r)
    psd_mat = np.empty((x_dim + 1, x_dim + 1), dtype=object)
    psd_mat[0, 0] = c + r
    psd_mat[0, 1:] = b.T / 2
    psd_mat[1:, 0] = b / 2
    psd_mat[1:, 1:] = S
    prog.AddPositiveSemidefiniteConstraint(psd_mat)
    utils.add_log_det_lower(prog, S, lower=0.0)
    return r


def is_ellipsoid_contained(
    S_inner: np.ndarray,
    b_inner: np.ndarray,
    c_inner: float,
    S_outer: np.ndarray,
    b_outer: np.ndarray,
    c_outer: float,
) -> bool:
    """
    Check if {x | x' * S_inner * x + b_inner' * x + c_inner <= 0} is contained
    in {x | x' * S_outer * x + b_outer' * x + c_outer <= 0}.
    """

    prog = solvers.MathematicalProgram()
    gamma = prog.NewContinuousVariables(1, "gamma")[0]

    # -x' * S_outer * x - b_outer' * x - c_outer
    #  + gamma * (x' * S_inner * x + b_inner' * x + c_inner) is sos.
    dim = S_inner.shape[0]
    mat = np.empty((dim + 1, dim + 1), dtype=object)
    mat[:dim, :dim] = gamma * S_inner - S_outer
    mat[:dim, -1] = (gamma * b_inner - b_outer) / 2
    mat[-1, :dim] = mat[:dim, -1]
    mat[-1, -1] = gamma * c_inner - c_outer
    prog.AddPositiveSemidefiniteConstraint(mat)
    result = solvers.Solve(prog)
    return result.is_success()


def add_ellipsoid_contain_pts_constraint(
    prog: solvers.MathematicalProgram,
    S: np.ndarray,
    b: np.ndarray,
    c: sym.Variable,
    pts: np.ndarray,
) -> solvers.Binding[solvers.LinearConstraint]:
    """
    Add the constraint that the ellipsoid {x | x'*S*x + b' * x + c <= 0} contains pts[i]

    Args:
      pts: pts[i] is the i'th point to be contained in the ellipsoid.
    """
    dim = S.shape[0]
    assert pts.shape[1] == dim
    x = sym.MakeVectorContinuousVariable(dim, "x")
    poly = sym.Polynomial(x.dot(S @ x) + b.dot(x) + c, sym.Variables(x))
    linear_coeffs, vars, constant = poly.EvaluateWithAffineCoefficients(x, pts.T)
    constraint = prog.AddLinearConstraint(
        linear_coeffs, np.full((pts.shape[0],), -np.inf), -constant, vars
    )
    return constraint


def scale_ellipsoid(S: np.ndarray, b: np.ndarray, c: float, scale: float) -> float:
    """
    Scale the "radius" of the ellipsoid {x | x'*S*x + b'*x + c<=0} about its
    center by a factor of `scale`.

    Return: c_new The scaled ellipsoid is {x | x'*S*x + b'*x + c_new <= 0}
    """
    assert scale >= 0

    # S and b are unchanged after scaling.
    # The ellipsoid can be written as |Ax + A⁻ᵀb/2|² <= c−bᵀS⁻¹b/4
    # where AᵀA=S
    # So c_new -bᵀS⁻¹b/4= scale² * (c−bᵀS⁻¹b/4)
    # namely c_new = scale² * c + (1-scale²)*bᵀS⁻¹b/4
    scale_squared = scale**2
    return scale_squared * c + (1 - scale_squared) * b.dot(np.linalg.solve(S, b)) / 4


def in_ellipsoid(
    S: np.ndarray, b: np.ndarray, c: float, pts: np.ndarray
) -> Union[bool, List[bool]]:
    """
    Return if `pts` is/are in the ellipsoid {x | x'*S*x+b'*x+c <= 0}.

    Args:
      pts: A single point or a group of points. Each row is a point.
    Return:
      flag: If pts is a 1-D array, then we return a single boolean. If pts is a
        2D array, then flag[i] indicates whether pts[i] is in the ellipsoid.
    """
    dim = S.shape[0]
    if pts.shape == (dim,):
        return pts.dot(S @ pts) + b.dot(pts) + c <= 0
    else:
        assert pts.shape[1] == dim
        return (np.sum(pts * (pts @ S), axis=1) + pts @ b + c <= 0).tolist()


def to_affine_ball(
    S: np.ndarray, b: np.ndarray, c: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the ellipsoid in the form {x | xᵀSx+bᵀx+c ≤ 0} to an alternative
    form as an affine transformation of a ball {A(y+d) | |y|₂ ≤ 1 }

    Args:
      S: A symmetric matrix that parameterizes the ellipsoid.
      b: A vector that parameterizes the ellipsoid.
      c: A float that parameterizes the ellipsoid.
    Returns:
      A: The affine transformation of the ball.
      d: The center of the ellipsoid.
    """

    # The ellipsoid can be written as
    # {x | xᵀA⁻ᵀA⁻¹x − 2dᵀA⁻¹x + dᵀd−1 ≤ 0}. To match it with xᵀSx+bᵀx+c ≤ 0,
    # we introduce a scalar k, with the constraint
    # A⁻ᵀA⁻¹ = kS
    # −2A⁻ᵀd = kb
    # dᵀd−1 = kc
    # To solve these equations, I define A̅ = A * √k, d̅ = d/√k
    # Hence I know A̅⁻ᵀA̅⁻¹ = S
    bar_A_inv = np.linalg.cholesky(S).T
    bar_A = np.linalg.inv(bar_A_inv)
    # −2A̅⁻ᵀd̅=b
    bar_d = bar_A.T @ b / -2
    # kd̅ᵀd̅−1 = kc
    k = 1 / (bar_d.dot(bar_d) - c)
    sqrt_k = np.sqrt(k)
    A = bar_A / sqrt_k
    d = bar_d * sqrt_k
    return (A, d)


def draw_ellipsoid2d(
    ax: matplotlib.axes.Axes, A: np.ndarray, d: np.ndarray, **kwargs
) -> List[matplotlib.lines.Line2D]:
    """
    Draw a 2D ellipsoid {A(y+d) | |y|₂ ≤ 1 }
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    xy_pts = A @ (
        np.concatenate([np.cos(theta).reshape((1, -1)), np.sin(theta).reshape((1, -1))])
        + d.reshape((-1, 1))
    )
    return ax.plot(xy_pts[0], xy_pts[1], **kwargs)


class Ellipsoid:
    """
    An ellipsoid as
    {x | xᵀSx+bᵀx+c≤0}
    """

    def __init__(self, S: np.ndarray, b: np.ndarray, c: float):
        assert S.shape[0] == S.shape[1]
        assert b.shape == (S.shape[0],)
        self.S = (S + S.T) / 2
        self.b = b
        self.c = c

    def to_affine_ball(self) -> Tuple[np.ndarray, np.ndarray]:
        return to_affine_ball(self.S, self.b, self.c)
