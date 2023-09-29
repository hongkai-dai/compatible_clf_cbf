from typing import Union
import numpy as np
import numpy.typing as npt

import pydrake.symbolic as sym


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
