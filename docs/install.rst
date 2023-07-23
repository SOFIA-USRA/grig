============
Installation
============

Stable release
--------------

The `grig` package is available via pip::

   pip install grig


From source
-----------

Obtain the source code for this package from the `Grig GitHub project
<https://github.com/SOFIA-USRA/grig>`__, then install via one of the
two methods below.

Via Anaconda
^^^^^^^^^^^^

We recommend Anaconda for managing your Python environment.  A conda
environment specification is included with this package, as
`environment.yml <https://raw.githubusercontent.com/SOFIA-USRA/grig/main/environment.yml>`__.

To install a ``grig`` environment with Anaconda from the package directory::


   conda env create -f environment.yml


Activate the environment::

   conda activate grig


Install an editable copy of the `grig` package::

   pip install -e .


Deactivate the environment when done::

   conda deactivate


Remove the environment if necessary::

   conda remove --name grig --all


Via Pip
^^^^^^^

Alternately, prerequisites for the package can be installed with::

  pip install -r requirements.txt

and the package can then be installed as usual::

   pip install -e .


Troubleshooting
---------------

Please note that direct support for this project will end in September 2023.

Prior to this time, please submit a ticket on the GitHub issues page for
installation assistance.

After this time, the source distribution of this package will remain available,
but it will not be maintained for the latest versions of all dependencies. It
is recommended that users fork their own copies of this package for continued
maintenance and development.

The last working set of installed versions of all dependencies is recorded in the
`freeze_requirements.txt <https://raw.githubusercontent.com/SOFIA-USRA/grig/main/freeze_requirements.txt>`__
file in this package. If errors are encountered in the other listed installation
methods, it may be useful to install the frozen versions directly. For example, to install
from source using conda to create a new Python environment from the grig package
directory::

   conda create --name grig python=3.11
   pip install -r freeze_requirements.txt
   pip install -e .


