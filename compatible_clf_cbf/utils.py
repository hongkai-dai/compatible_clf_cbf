from typing import Optional, Union
import numpy as np

import pydrake.symbolic as sym
import pydrake.solvers as solvers


def check_array_of_polynomials(p: np.ndarray, x_set: sym.Variables) -> None:
    """
    Check if each element of p is a symbolic polynomial, whose indeterminates
    are a subset of `x_set`.
    """
    assert isinstance(p, np.ndarray)
    assert isinstance(x_set, sym.Variables)

    for p_i in p.flat:
        assert isinstance(p_i, sym.Polynomial)
        if not p_i.indeterminates().IsSubsetOf(x_set):
            raise Exception(f"{p_i}'s indeterminates is not a subset of {x_set}")


def check_polynomial_arrays_equal(p: np.ndarray, q: np.ndarray, tol: float):
    assert p.shape == q.shape
    for p_i, q_i in zip(p.flat, q.flat):
        assert p_i.CoefficientsAlmostEqual(q_i, tol)


def get_polynomial_result(
    result: solvers.MathematicalProgramResult,
    p: Union[sym.Polynomial, np.ndarray],
    coefficient_tol: Optional[float] = None,
) -> Union[sym.Polynomial, np.ndarray]:
    """
    Given a MathematicalProgramResult and an array of symbolic Polynomials
    (or a single symbolic Polynomial), return the result of these symbolic
    Polynomials. Remove the terms in the polynomials if the absolute vlues of
    the coefficients are <= coefficient_tol.
    """
    if isinstance(p, sym.Polynomial):
        p_result = result.GetSolution(p)
        if coefficient_tol is not None:
            return p_result.RemoveTermsWithSmallCoefficients(coefficient_tol)
        else:
            return p_result
    else:
        p_result = np.array([result.GetSolution(p_i) for p_i in p.flat]).reshape(
            p.shape
        )
        if coefficient_tol is not None:
            p_result = np.array(
                [
                    p_result_i.RemoveTermsWithSmallCoefficients(coefficient_tol)
                    for p_result_i in p_result.flat
                ]
            ).reshape(p.shape)
        return p_result


def is_sos(p: sym.Polynomial) -> bool:
    prog = solvers.MathematicalProgram()
    prog.AddIndeterminates(p.indeterminates())
    prog.AddSosConstraint(p)
    result = solvers.Solve(prog)
    return result.is_success()
