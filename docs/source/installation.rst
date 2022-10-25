Installation
############

This `py-modelrunner` package is developed for python 3.8+ and should run on all common
platforms. 



Install using pip
^^^^^^^^^^^^^^^^^

The package is available on `pypi <https://pypi.org/project/py-modelrunner/>`_, so you
should be able to install it by running

.. code-block:: bash

    pip install py-modelrunner

Install using conda
^^^^^^^^^^^^^^^^^^^

The `py-modelrunner` package is also available on `conda <https://conda.io>`_ using the
`conda-forge` channel.
You can thus install it using

.. code-block:: bash

    conda install -c conda-forge py-modelrunner

This installation includes all required dependencies to have all features of `py-pde`.


Installing from source
^^^^^^^^^^^^^^^^^^^^^^


Prerequisites
-------------

The code builds on other python packages, which need to be installed for
this package to function properly.
The required packages are listed in the table below:

===========  ========= =========
Package      Version   Usage 
===========  ========= =========
jinja2       >=2.7     Dealing with templates for launching simulations
h5py         >=3.5     Storing data in the HDF format 
numpy        >=1.18.0  Array library used for storing data
pandas       >=1.3     Data tables for structured data access
PyYAML       >=5       Storing data in the YAML format
tqdm         >=4.45    Show progress bar
===========  ========= =========

These package can be installed via your operating system's package manager, e.g.
using :command:`conda`, or :command:`pip`.
The package versions given above are minimal requirements, although
this is not tested systematically. Generally, it should help to install the
latest version of the package.  


Downloading the package
-----------------------

The package can be simply checked out from
`github.com/zwicker-group/py-modelrunner <https://github.com/zwicker-group/py-modelrunner>`_.
To import the package from any python session, it might be convenient to include
the root folder of the package into the :envvar:`PYTHONPATH` environment variable.

This documentation can be built by calling the :command:`make html` in the
:file:`docs` folder.
The final documentation will be available in :file:`docs/build/html`.
Note that a LaTeX documentation can be build using :command:`make latexpdf`.
