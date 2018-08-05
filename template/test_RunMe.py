"""
Warning: LONG RUNTIME TESTS!

This test suite is designed to verify that the main components of the framework are not broken.
It is expected that smaller components or sub-parts are tested individually.

As we all know this will probably never happen, we will at least verify that the overall features
are correct and fully functional. These tests will take long time to run and are not supposed to
be run frequently. Nevertheless, it is important that before a PR or a push on the master branch
the main functions can be tested.

Please keep the list of these tests up to date as soon as you add new features.
"""
import numpy as np

from template.RunMe import RunMe

def test_one():
    """
    - Verify the sizes of the return of execute
    - Image classification with default parameters
    """
    epochs = 1
    args = ["--experiment-name", "test_template",
            "--ignoregit",
            "--dataset-folder", "/dataset/cifar",
            "--epochs", str(epochs)]

    train, val, test = RunMe().main(args=args)

    # Verify sizes of the return values from execute
    assert len(train.shape) == 2
    assert train.shape[0] == 1
    assert train.shape[1] == epochs

    assert len(val.shape) == 2
    assert train.shape[0] == 1
    assert train.shape[1] == epochs+1

    assert len(train.shape) == 1
    assert train.shape[0] == 1

    # Verify values
    np.testing.assert_almost_equal(train[0], 100.0)
    np.testing.assert_almost_equal(val[0], 100.0)
    np.testing.assert_almost_equal(test, 100.0)
