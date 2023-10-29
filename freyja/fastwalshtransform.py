import numpy as np


def is_power_of_two(num):
    """
    Check if num is power of 2
    """
    return num > 0 and ((num & (num - 1)) == 0)


def ceil_power_of_two(num):
    """
    Rounds num up to the next power of 2
    """
    x = 1
    while x < num:
        x *= 2

    return x


def fast_walsh_transform(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Peform the fast Walsh transform on the input signal
    and return the Walsh coefficients,
    see Beer, Am. J. Phys 49 (1981).

    Parameters
    ----------
    x: np.ndarray
        The input signal to be transformed. Will be zero padded if length is not a power of two.
    normalize: bool
        Whether to devide the transform by the square root of the zero-padded signal length.
        Defaults to True.

    Return
    ------
    np.ndarray
        The Walsh transform of the input signal.
    """
    shape = x.shape
    length = shape[-1]
    # Check if N is a power of 2
    if not is_power_of_two(shape[-1]):
        length = ceil_power_of_two(shape[-1])

    x_copy = np.zeros((shape[:-1] + (length,)))
    x_copy[..., : shape[-1]] = x
    step = 1

    a = np.zeros((shape[:-1] + (length,)))
    b = np.zeros((shape[:-1] + (length,)))

    sign = np.asarray([(-1) ** n for n in range(length)])

    while step < length:
        for i in range(0, length, 2 * step):
            a[..., i : i + 2 * step] = np.repeat(x_copy[..., i : i + step], 2, axis=-1)
            b[..., i : i + 2 * step] = np.repeat(
                sign[:step] * x_copy[..., i + step : i + 2 * step], 2, axis=-1
            )
        x_copy = a + sign * b

        step *= 2

    if normalize is True:
        x_copy = x_copy / np.sqrt(length)

    return x_copy
