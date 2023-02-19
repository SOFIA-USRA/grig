# Licensed under a 3-clause BSD style license - see LICENSE.rst

import gc
import sys

import numpy as np

__all__ = ['slicer', 'taylor', 'julia_fractal', 'byte_size_of_object']


def slicer(array, axis, index, ind=False):
    """
    Returns a slice of an array in arbitrary dimension.

    Parameters
    ----------
    array : numpy.ndarray
        array to slice
    axis : int or array_like
        axis to slice on
    index : int or array_like of int
        index retrieved
    ind : bool, optional
        If True, return the slices rather than sliced array

    Returns
    -------
    numpy.ndarray or tuple of slice
    """
    if isinstance(index, int):
        idx = [slice(None)] * axis
        idx += [index]
        idx += [slice(None)] * (array.ndim - axis - 1)
        idx = tuple(idx)
    else:
        idx = list(index)
        idx.insert(axis, slice(None))
        idx = tuple(idx)

    if ind:
        return idx
    else:
        return array[idx]


def taylor(order, n):
    """
    Taylor expansion generator for Polynomial exponents

    Parameters
    ----------
    order : int
        Order of Polynomial
    n : int
        Number of variables to solve for

    Yields
    ------
    n-tuple of int
        The next polynomial exponent
    """
    if n == 0:
        yield ()
        return
    for i in range(order + 1):
        for result in taylor(order - i, n - 1):
            yield (i,) + result


def julia_fractal(sy, sx, c0=-0.4, c1=0.6, iterations=256,
                  xrange=(-1, 1), yrange=(-1, 1), normalize=True):
    """
    Generate a 2-D Julia fractal image

    Parameters
    ----------
    sy : int
        y dimension size.
    sx : int
        x dimension size.
    c0 : float, optional
        The c0 coefficient.
    c1 : float, optional
        The c1 coefficient.
    iterations : int, optional
        The number of steps.
    xrange : array_like of int or float, optional
        The range of x values.
    yrange : array_like of int or float, optional
        The range of y values.
    normalize : bool, optional

    Returns
    -------

    """
    x = np.linspace(xrange[0], xrange[1], sx)[None]
    y = np.linspace(yrange[0], yrange[1], sy)[..., None]
    z = np.tile(x, (sy, 1)) + 1j * np.tile(y, (1, sx))
    c = c0 + 1j * c1
    mask = np.full((sy, sx), True)
    result = np.zeros((sy, sx))

    for i in range(iterations):
        z[mask] *= z[mask]
        z[mask] += c
        mask[np.abs(z) > 2] = False
        result[mask] = i

    if normalize:
        result /= result.max()

    return result


def byte_size_of_object(obj):
    """
    Return the size of a Python object in bytes.

    Parameters
    ----------
    obj : object

    Returns
    -------
    byte_size : int
    """
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr
                    if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz
