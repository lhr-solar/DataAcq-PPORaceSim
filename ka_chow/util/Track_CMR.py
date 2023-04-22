import splines
import numpy as _np
from splines import _check_param


class Track_CMR(splines.ConstantSpeedAdapter):
    def __init__(self, curve):
        super().__init__(curve)

    def evaluate(self, s, n=0):
        if not _np.isscalar(s):
            return _np.array([self.evaluate(s, n) for s in s])
        return self.curve.evaluate(self._s2t(s), n)
