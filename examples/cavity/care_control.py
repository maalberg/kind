# --! utility functions for cavity resonance control --!


import numpy as np
from scipy import signal
from scipy import linalg
from collections import namedtuple


matrices_elec = namedtuple('matrices_elec', 'a b')
matrices_mech = namedtuple('matrices_mech', 'a b c d')


def create_matrices_elec(f, q):
    """
    Creates continuous-time electrical matrices A and B.

    Parameters:
        f: float, operating frequency of radio frequency signal, Hz
        q: float, loaded quality factor of cavity, unitless

    Returns:
        mat: tuple of 2D numpy arrays, matrices A and B that describe the electrical cavity equation
    """

    a = create_mat_ae(f, q)
    b = create_mat_be(f, q)

    return matrices_elec(a, b)


def create_matrices_mech(f, q, k, discrete=False, dt=0.0):
    """
    Creates state-space mechanical matrices A, B, C and D.

    Parameters:
        f: 1D numpy array of floats, frequencies of mechanical modes, Hz
        q: 1D numpy array of floats, quality factors of mechanical modes, unitless
        k: 1D numpy array of floats, couplings of mechanical modes, rad/s / (MV)^2
        discrete: boolean flag, whether or not to discretize created matrices, [default=False]
        dt: float, discretization step, s [default=0]

    Returns:
        mat: tuple of 2D numpy arrays, matrices A, B, C and D that describe the mechanical cavity equation
    """

    a = create_mat_am(f, q)
    b = create_mat_bm(f, k)
    c = create_mat_cm(len(f))
    d = create_mat_dm()

    if discrete:
        a, b, c, d, _ = signal.cont2discrete((a, b, c, d), dt)

    return matrices_mech(a, b, c, d)


def create_mat_ae(f, q):
    """
    Creates matrix A for electrical cavity equation.

    Parameters:
        f: float, operating frequency of cavity drive signal, Hz
        q: float, loaded quality factor of cavity, unitless

    Returns:
        ae: 2D numpy array, matrix A for electrical cavity equation
    """
    hbw = np.pi * f / q  # cavity half-bandwidth in rad/s

    return np.array([
        [ -hbw,     0. ],
        [  0.,    -hbw ],
    ])


def create_mat_be(f, q):
    """
    Creates matrix B for electrical cavity equation.

    Parameters:
        f: float, operating frequency of cavity drive signal, Hz
        q: float, loaded quality factor of cavity, unitless

    Returns:
        be: 2D numpy array, matrix B for electrical cavity equation
    """
    hbw = np.pi * f / q  # cavity half-bandwidth in rad/s

    return np.array([
        [ hbw,    0. ],
        [ 0.,    hbw ],
    ])


def create_mat_am(f, q):
    """
    Creates matrix A for mechanical cavity equation.

    Parameters:
        f: 1D numpy array of floats, frequencies of mechanical modes, Hz
        q: 1D numpy array of floats, quality factors of mechanical modes, unitless

    Returns:
        am: 2D numpy array, matrix A for mechanical cavity equation that accommodates all modes
    """

    def create_mat_a(f, q):
        w = 2 * np.pi * f
        return np.array([
            [  0,             1   ],
            [ -np.square(w), -w/q ],
        ])

    return linalg.block_diag(*[create_mat_a(frequency, quality) for frequency, quality in zip(f, q)])


def create_mat_bm(f, k):
    """
    Creates matrix B for mechanical cavity equation.

    Parameters:
        f: 1D numpy array of floats, frequencies of mechanical modes, Hz
        k: 1D numpy array of floats, couplings of mechanical modes, rad/s / (MV)^2

    Returns:
        bm: 2D numpy array, matrix B for mechanical cavity equation that accommodates all modes
    """

    def create_mat_b(f, k):
        w = 2 * np.pi * f
        return np.array([
            [  0                ],
            [ -k * np.square(w) ],
        ])

    return np.concatenate([create_mat_b(frequency, coupling) for frequency, coupling in zip(f, k)], axis=0)


def create_mat_cm(nm):
    """
    Creates matrix C for mechanical cavity equation.

    Parameters:
        nm: integer, number of mechanical modes

    Returns:
        cm: 2D numpy array, matrix C for mechanical cavity equation that accommodates all modes
    """

    def create_mat_c():
        return np.array([[1, 0]])

    return np.tile(create_mat_c(), nm)


def create_mat_dm():
    """
    Creates matrix D for mechanical cavity equation.

    Returns:
        dm: 2D numpy array, matrix D for mechanical cavity equation
    """
    return np.array(([[0]]))

