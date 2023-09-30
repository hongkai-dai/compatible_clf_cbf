import compatible_clf_cbf.clf_cbf as mut

import numpy as np
import pytest

import pydrake.symbolic as sym


class TestClfCbf(object):
    @classmethod
    def setup_class(self):
        self.x = sym.MakeVectorContinuousVariable(3, "x")
        # The dynamics and unsafe regions are arbitrary, they are only used to
        # test the functionality of the code.
        self.f = np.array(
            [
                sym.Polynomial(self.x[0] * self.x[0]),
                sym.Polynomial(self.x[1] * self.x[0]),
                sym.Polynomial(self.x[0] + self.x[2]),
            ]
        )
        self.g = np.array(
            [
                [sym.Polynomial(self.x[0] + self.x[1]), sym.Polynomial()],
                [sym.Polynomial(), sym.Polynomial(self.x[1] * self.x[2])],
                [sym.Polynomial(self.x[0] + self.x[2]), sym.Polynomial(self.x[1])],
            ]
        )
        self.unsafe_regions = [
            np.array([sym.Polynomial(self.x[0] + 1)]),
            np.array([sym.Polynomial(1 - self.x[1]), sym.Polynomial(1 - self.x[0])]),
        ]

    def test_constructor(self):
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=True,
            use_y_squared=True,
        )
        assert dut.y.shape == (len(self.unsafe_regions) + 1,)

        # Now construct with with_clf=False
        dut = mut.CompatibleClfCbf(
            f=self.f,
            g=self.g,
            x=self.x,
            unsafe_regions=self.unsafe_regions,
            Au=None,
            bu=None,
            with_clf=False,
            use_y_squared=True,
        )
        assert dut.y.shape == (len(self.unsafe_regions),)
