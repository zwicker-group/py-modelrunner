User Manual
===========

Main structure of the package
-----------------------------

The package offers several different components that can be used separately or together:

Hierarchical input/output using :mod:`~modelrunner.storage`:
    The module provides an abstract interface for writing and reading data using an
    hierarchical organization.
    Storages are conveniently created by :func:`~modelrunner.storage.tools.open_storage`,
    which returns a storage object that offers a dict-like interface.

Defining parameters using :mod:`~modelrunner.model.parameters`:
    The module provides the :class:`~modelrunner.model.parameters.Parameter` class, describing a single parameter. 
    This can be used together with the mixin :class:`~modelrunner.model.parameters.Parameterized`,
    which allows to equip classes with default parameters and some convenience methods.

Defining models using :mod:`~modelrunner.model`:
    Models are augmented functions, which define input, calculations, and output.
    Models can be conveniently created by decorating a function or by subclassing
    :class:`~modelrunner.model.base.ModelBase`, which is built on the parameter classes.

Model results are captured by :mod:`~modelrunner.run.results`:
    Results are returned as the special :class:`~modelrunner.run.results.Result`, which
    keeps track of the input parameters and the data calculated by the model.
    :class:`~modelrunner.run.results.ResultCollection` describes a collections of the same
    model evoked with different parameter values.

Submitting models to HPC using :mod:`~modelrunner.run`:
    A single model can be submitted to a compute node using :func:`~modelrunner.run.job.submit_job`,
    e.g., to run the computation on a high performance compute cluster.
    A parameter study using multiple jobs can be conveniently submitted using :func:`~modelrunner.run.job.submit_jobs`.
    The results written to one directory can then be conveniently analyzed using :func:`~modelrunner.run.results.ResultCollection.from_folder`.


Design philosophy
-----------------

The main requirements for the package can be summarized as follows:

- **Usability**: The user should not need to think about how data is stored in different files. The :class:`~modelrunner.run.results.Result` class should simply work.
- **Flexibility**: :mod:`~modelrunner.storage` should provide a unified interface to write data in multiple file formats (JSON, YAML, HDF, zarr, ...)
- **Stability**: Future versions of the package should be able to read older files even when the internal definitions of file formats change
- **Modularity**: Different parts of the package (like :mod:`~modelrunner.storage`, :mod:`~modelrunner.model.parameters`, and :mod:`~modelrunner.run`) should be rather independent of each other, so they can be used in isolation
- **Extensibility**: Models inherting from :class:`~modelrunner.model.base.ModelBase` should be easy to subclass to implement more complicated requirements (e.g., additional parameters)
- **Self-explainability**: The files should in principle contain all information to reconstruct the data, even if the :mod:`modelrunner` package is no longer available.
- **Efficiency**: The files should only store necessary information.

The last point results in particular constraints if we want to store temporal simulation results.
In most cases, there are are some data that are kept fixed for the simulation (describing physical parameters) and others that evolve with time.
We denote by `attributes` the parameters that are kept fixed and by `data` the data that varies over time.
The :mod:`~modelrunner.storage.trajectory` module deals with such data.