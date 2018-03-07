import numpy as np
import sys
sys.path.append('../')
from freyja import fastwalshtransform #with __init__.py
#from freyja.fastwalshtransform import fastwalshtransform


def test_fastwalshtransform_1():
    """
    Transforms a vector and compares it to known transformation.
    """
    x=np.array([1,2,3,4])
    y=np.array([5,-2,0,-1])
    z=fastwalshtransform(x)
    np.testing.assert_array_equal(y, z)


def test_fastwalshtransform_2():
    """
    Foward and backward transformation of a random vector.
    """
    x=np.random.rand(32)
    y=fastwalshtransform(x)
    z=fastwalshtransform(y)
    np.testing.assert_allclose(x, z)
