import math

import numpy as np

from util.evaluation.metrics import _apk


def test_apk():
    assert math.isnan(_apk(1, [0, 0]))
    assert _apk(1, [0, 1], 1) == 0.0
    assert _apk(1, [1, 0]) == 1.0
    assert _apk(1, [0, 1], 2) == 0.5
    assert _apk(1, [0, 1, 0, 1], 'full') == 0.5
    assert _apk(1, [0, 1, 0, 1], 'auto') == 0.25
    assert _apk(1, [0, 0, 1, 1], 'auto') == 0.0
    np.testing.assert_almost_equal(_apk(1, [0, 0, 1, 1], 'full'), 0.41666666)
    assert _apk(1, [0, 0, 1, 0, 0, 1], 'full') == 1.0 / 3
    np.testing.assert_almost_equal(_apk(1, [1, 0, 1, 0, 0, 1, 0, 0, 1, 1], 'full'), 0.62222222222)
    assert _apk(1, [1] * 10000000, 'full') == 1.0
