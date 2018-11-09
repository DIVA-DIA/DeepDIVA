import numpy as np
import pytest

from util.evaluation.metrics import apk, mapk


def test_sanity_check():
    # Test k > |prediction|
    with pytest.raises(AssertionError):
        apk(1, [1, 2, 3], 8)

    # Test k == 0
    with pytest.raises(AssertionError):
        apk(1, [1, 2, 3], 0)

    # Test k non int
    with pytest.raises(AssertionError):
        apk(1, [1, 2, 3], 'hello')

    # Test []
    with pytest.raises(AssertionError):
        apk(1, [])

def test_apk_base_case():
    # Test base base case scenario
    assert apk(1, [0, 0]) == 0.0
    assert apk(1, [0, 1]) == 0.5
    assert apk(1, [1, 0]) == 1.0
    assert apk(1, [1, 1]) == 1.0


def test_apk_full():
    # Test non-zero labels & 'full'
    assert apk(1, [2, 3], 'full') == 0.0
    assert apk(1, [4, 1], 'full') == 0.5
    assert apk(1, [1, 5], 'full') == 1.0
    assert apk(1, [1, 1], 'full') == 1.0


def test_apk_atk():
    # Test @k
    assert apk(1, [0, 0], 1) == 0.0
    assert apk(1, [0, 1], 1) == 0.0
    assert apk(1, [1, 0], 1) == 1.0
    assert apk(1, [1, 1], 1) == 1.0


def test_apk_auto():
    # Test 'auto'
    assert apk(1, [0, 0], 'auto') == 0.0
    assert apk(1, [0, 1], 'auto') == 0.0
    assert apk(1, [1, 0], 'auto') == 1.0
    assert apk(1, [1, 1], 'auto') == 1.0


def test_apk_advanced():
    # Test with more than 2 numbers
    assert apk(1, [0, 1, 0, 1]) == 0.5
    assert apk(1, [0, 1, 0, 1], 2) == 0.25
    assert apk(1, [0, 1, 0, 1], 'auto') == 0.25

    assert apk(1, [2, 3, 1, 1], 1) == 0.0
    assert apk(1, [2, 3, 1, 1], 2) == 0.0
    assert apk(1, [2, 3, 1, 1], 'auto') == 0.0
    np.testing.assert_almost_equal([apk(1, [2, 3, 1, 1], 3)], [0.16666666])
    np.testing.assert_almost_equal([apk(1, [2, 3, 1, 1], 4)], [0.41666666])
    np.testing.assert_almost_equal([apk(1, [2, 3, 1, 1], 'full')], [0.41666666])


def test_apk_long_sequences():
    # Test longer sequences
    assert apk(1, [1] * 15000) == 1.0


def test_mapk():
    # Single entry
    np.testing.assert_almost_equal([mapk([1], [[2, 3, 1, 1]], 'full')[0]], [0.41666666])

    # Multiple entries
    np.testing.assert_almost_equal([mapk([1, 1], [[2, 3, 1, 1], [1, 1]], 'full')[0]], [0.70833333])
