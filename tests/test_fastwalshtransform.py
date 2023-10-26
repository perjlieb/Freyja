"""
Testing the fast Walsh transform
"""
import pytest
import numpy as np

from freyja import fast_walsh_transform


@pytest.mark.parametrize(
    "test_input, expected", [(np.array([1, 2, 3, 4]), np.array([5, -2, 0, -1]))]
)
def test_fastwalshtransform(test_input, expected):
    """
    Transforms a vector and compares it to known transformation.
    """
    z = fast_walsh_transform(test_input)
    np.testing.assert_array_equal(expected, z)


def test_fastwalshtransform_back_and_forth():
    """
    Foward and backward transformation of a random vector.
    """
    x = np.random.rand(32)
    y = fast_walsh_transform(x)
    z = fast_walsh_transform(y)
    np.testing.assert_allclose(x, z)
