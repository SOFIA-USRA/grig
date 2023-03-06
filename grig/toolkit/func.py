# Licensed under a 3-clause BSD style license - see LICENSE.rst

import gc
import sys
import warnings

import bottleneck as bn
import numpy as np
from scipy.stats import describe

__all__ = ['slicer', 'taylor', 'julia_fractal',
           'byte_size_of_object', 'robust_mask', 'moments']


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


def robust_mask(data, threshold, mask=None, axis=None, mask_data=False,
                cval=np.nan):
    """
    Computes a mask derived from data Median Absolute Deviation (MAD).

    Calculates a robust mask based on the input data and optional input mask.
    If :math:`threshold > 0`, the dataset is searched for outliers.  Outliers
    are identified for point :math:`i` if

    .. math::

        \\frac{|y_i - median[y]|}{MAD} > threshold

    where :math:`MAD` is the Median Absolute Deviation defined as

    .. math::

        MAD = 1.482 * median[|y_i - median[y]|]

    Parameters
    ----------
    data : array_like of float
        The data on which to derive a robust mask.
    threshold : float
        Threshold as described above.
    mask : array_like of bool, optional
        If supplied, must be the same shape as `data`.  Any masked (`False`)
        `data` values will not be included in the :math:`MAD` calculation.
        Additionally, masked elements will also be masked (`False`) in the
        output mask.
    axis : int, optional
        Axis over which to calculate the :math:`MAD`.  The default (`None`)
        derives the :math:`MAD` from the entire set of `data`.
    mask_data : bool, optional
        If `True`, return a copy of `data` with masked values replaced by
        `cval` in addition to the output mask.  The default is `False`.
        Note that the output type will
    cval : int or float, optional
        if `mask_data` is set to `True`, masked values will be replaced by
        `cval`.  The default is `numpy.nan`.

    Returns
    -------
    numpy.ndarray of bool, [numpy.ndarray of numpy.float64]
        The output mask where `False` indicates a masked value, while `True`
        indicates that associated data deviation is below the `threshold`
        limit.  If `mask_data` was `True`, also returns a copy of `data`
        with masked values replaced by `cval`.
    """
    d = np.asarray(data, dtype=float).copy()
    valid = np.isfinite(d)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != d.shape:
            raise ValueError("data and mask shape mismatch")
        valid &= mask

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', FutureWarning)
        if threshold is not None and threshold > 0:
            d[~valid] = np.nan
            if axis is None:
                med = bn.nanmedian(d)
                mad = 1.482 * bn.nanmedian(np.abs(d - med))
            else:
                med = np.expand_dims(bn.nanmedian(d, axis=axis), axis)
                mad = np.expand_dims(
                    1.482 * bn.nanmedian(
                        np.abs(d - med), axis=axis), axis)

            ratio = np.abs(d - med) / mad
            valid &= ratio <= threshold

    if mask_data:
        d[~valid] = cval
        return valid, d
    else:
        return valid


def moments(data, threshold=None, mask=None, axis=None, get_mask=False):
    """
    Computes statistics on a data set avoiding deviant points if requested

    Moments are calculated for a given set of data.  If a value is passed
    to threshold, then the dataset is searched for outliers.  A data point
    is identified as an outlier if abs(x_i - x_med)/MAD > threshold, where
    x_med is the median, MAD is the median absolute deviation defined as
    1.482 * median(abs(x_i - x_med)).

    Parameters
    ----------
    data : array_like of float
        (shape1) Data on which to calculate moments
    mask : array_like of bool
        (shape1) Mask to apply to data
    threshold : float, optional
        Sigma threshold over which values are identified as outliers
    axis : int, optional
        Axis over which to calculate statistics
    get_mask : bool, optional
        If True, only return the output mask

    Returns
    -------
    dict or numpy.ndarray
        If `get_mask` is False, returns a dictionary containing the
        following statistics: mean, var, stddev, skew, kurt, stderr,
        mask.  Otherwise, returns the output mask.
    """

    valid = robust_mask(data, threshold, mask=mask, axis=axis,
                        mask_data=np.logical_not(get_mask), cval=np.nan)

    if get_mask:
        return valid
    else:
        valid, d = valid

    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', FutureWarning)
        stats = describe(d, nan_policy='omit', axis=axis)

    result['mask'] = valid
    result['mean'] = getattr(stats, 'mean')
    result['var'] = getattr(stats, 'variance')
    result['stddev'] = np.sqrt(result['var'])
    result['skewness'] = getattr(stats, 'skewness')
    result['kurtosis'] = getattr(stats, 'kurtosis')
    for k, v in result.items():
        if isinstance(v, np.ma.MaskedArray):
            result[k] = v.data

    return result
