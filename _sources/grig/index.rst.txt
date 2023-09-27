******************
Grig Documentation
******************

Grig is a fully N-dimensional resampling package for irregularly
gridded data with associated errors. It was primarily developed for
generating astronomical image mosaics and spectral image cubes, but has broad
applicability to scientific data arrays with or without associated errors.

Given a cloud of irregularly spaced data points, the resampling algorithm
assigns values to voxels of a regular grid by fitting data points in a local cloud
with a low-order polynomial function, by default. See the figure below for an
illustration.

The resampling algorithm has a wide set of parameters to allow customization to
particular use cases:

    - Splines can be used in place of polynomial fits for the local interpolation.
    - The convolution kernel may be regular or irregular.
    - The kernel size and shape may be fixed or may adapt to the data.

See the :ref:`example section <resample_examples>` for more possible applications, or the
`white paper <https://raw.githubusercontent.com/SOFIA-USRA/grig/main/docs/grig/resampling/paper/Perera.pdf>`__
included with the source distribution for a complete technical description.

.. figure:: resampling/images/resample_algorithm.png
   :alt: Left: Observations: Cloud of irregular data points with spatial (x,y) and
         spectral (lambda) coordinates. The image shows many irregularly spaced points.
         Middle: 1. Model observations as F = f(x, y, lambda) + epsilon where F = flux,
         epsilon = error, and f is a polynomial function. 2. For voxel (i, j, k), derive
         the local function f by fitting to a localized cloud of samples in the region
         of (x-i, y-j, lambda-k).  3. Set F-i-j-k = f(x-i, y-i, lambda-k) where F-i-j-k
         is the flux at voxel (i, j, k).  The image shows a three dimensional grid containing
         a sphere, representing the local datacloud. The sphere contains a small red cube,
         representing a single output voxel.
         Right: Data Cube: Regularly spaced array of voxels where each voxel element
         is at coordinate (x-i, y-j, lambda-k) and i, j, k respectively mark indices
         along dimensions x, y, lambda of the cube.  The image shows a regular cube with
         edges labeled x, y, and lambda, containing a small red cube representing the
         voxel from the middle image.
   :name: resample_algorithm

   Resampling algorithm application to a spectral cube with two spatial dimensions and
   one spectral.  The algorithm assigns values to voxels of a regular grid by fitting data
   points in a local cloud with a low-order polynomial function.  Data points in the local
   cloud are weighted by distance from the output point and by their associated error
   estimates.


Getting Started
===============

.. include:: ../install.rst


Submodules
==========

.. toctree::
  :maxdepth: 2

  resampling/index.rst
  toolkit/index.rst

Reference/API
=============

.. toctree::
   ../source/grig
