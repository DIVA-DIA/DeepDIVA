import math

import numpy as np

from util.evaluation.metrics import _apk


def test_apk_corner_cases():
    # Sanity check
    assert _apk(1, []) == 0.0

    # Test k > |prediction|
    assert _apk(5, [5, 4, 3, 2, 1, 0], 8) == 0.125
    assert _apk(5, [5, 4, 3, 2, 1, 0], 10) == 0.1


def test_apk_base_case():
    # Test base base case scenario
    assert _apk(1, [0, 0]) == 0.0
    assert _apk(1, [0, 1]) == 0.25
    assert _apk(1, [1, 0]) == 0.5
    assert _apk(1, [1, 1]) == 1.0


def test_apk_full():
    # Test non-zero labels & 'full'
    assert _apk(1, [2, 3], 'full') == 0.0
    assert _apk(1, [4, 1], 'full') == 0.25
    assert _apk(1, [1, 5], 'full') == 0.5
    assert _apk(1, [1, 1], 'full') == 1.0


def test_apk_atk():
    # Test @k
    assert _apk(1, [0, 0], 1) == 0.0
    assert _apk(1, [0, 1], 1) == 0.0
    assert _apk(1, [1, 0], 1) == 1.0
    assert _apk(1, [1, 1], 1) == 1.0


def test_apk_auto():
    # Test 'auto'
    assert _apk(1, [0, 0], 'auto') == 0.0
    assert _apk(1, [0, 1], 'auto') == 0.0
    assert _apk(1, [1, 0], 'auto') == 1.0
    assert _apk(1, [1, 1], 'auto') == 1.0


def test_apk_advanced():
    # Test with more than 2 numbers
    assert _apk(1, [0, 1, 0, 1]) == 0.25
    assert _apk(1, [0, 1, 0, 1], 2) == 0.25
    assert _apk(1, [0, 1, 0, 1], 'auto') == 0.25

    assert _apk(1, [2, 3, 1, 1], 1) == 0.0
    assert _apk(1, [4, 5, 1, 1], 2) == 0.0
    assert _apk(1, [6, 7, 1, 1], 'auto') == 0.0
    np.testing.assert_almost_equal(_apk(1, [2, 3, 1, 1], 3), 0.11111111)
    np.testing.assert_almost_equal(_apk(1, [5, 4, 1, 1], 4), 0.20833333)
    np.testing.assert_almost_equal(_apk(1, [6, 6, 1, 1]), 0.20833333)
    np.testing.assert_almost_equal(_apk(1, [11, 12, 1, 1], 'full'), 0.20833333)


def test_apk_long_sequences():
    # Test longer sequences
    assert _apk(1, [1] * 10000) == 1.0

