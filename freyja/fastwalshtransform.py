import numpy as np

def ispoweroftwo(num):
    """
    Check if num is power of 2
    """
    return num > 0 and ((num & (num - 1)) == 0)


def ceilpoweroftwo(num):
    """
    Rounds num up to the next power of 2
    """
    x = 1
    while(x < num):
        x *= 2

    return x


def fastwalshtransform(x):
    """
    Peforms the fast Walsh transform on the input signal
    and returns the Walsh coefficients,
    see Beer, Am. J. Phys 49 (1981).
    x: input signal
    N: fwt size
    returns z: Walsh coefficients
    """
    N = x.size
    # Check if N is a power of 2
    if not(ispoweroftwo(N)):
        N = ceilpoweroftwo(N)

    dN = N - x.size
    x = np.append(x,np.zeros(dN))
    z = np.zeros(N)
    M = 1

    while M != N:
        points = range(0, N, 2*M)
        for i in points:
            for j in range(2*M):
                z[i+j] = x[i+(j//2)] + ((-1)**(j+(j//2))) * x[i+(j//2)+M]

        x[:] = z[:]
        M *= 2

    return z/np.sqrt(N)
