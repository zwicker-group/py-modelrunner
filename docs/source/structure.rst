Developer Manual
================

Main structure of the package
-----------------------------

The following graph summarizes the classes responsible for defining models and
corresponding states.

.. graphviz::

   digraph Structure {
      node [shape=box]  /* default attributes */

      IOBase [label=<<b>IOBase</b><br/>Basic input/output methods>, href="../packages/modelrunner.state.io.html#modelrunner.state.io.IOBase", target="_top"];
      State [label=<<b>State</b><br/>Defines simulation state:<br align="left"/>- data ... degrees of freedom<br align="left"/>- attributes ... additional information>, href="../packages/modelrunner.state.html", target="_top"];
      Trajectory [label=<<b>Trajectory</b><br/>States as a function of time>, href="../packages/modelrunner.state.trajectory.html", target="_top"];
      Model [label=<<b>Model</b><br/>Describing simulation dynamics<br align="left"/>- Contains model parameters<br align="left"/>>, href="../packages/modelrunner.model.html", target="_top"];
      Result [label=<<b>Result</b><br/>Combination of model and a state<br align="left"/>- Contains all information for further analysis>, href="../packages/modelrunner.results.html#modelrunner.results.Result", target="_top"];
      ResultCollection [label=<<b>ResultCollection</b><br/>Results from different model parameters<br align="left"/>- Deals with parameter sweeps<br align="left"/>>, href="../packages/modelrunner.results.html#modelrunner.results.ResultCollection", target="_top"];

      IOBase -> State -> Result;
      State -> Trajectory;
      Model -> Result;
      IOBase -> Result -> ResultCollection;
   }


Design philosophy
-----------------

The main requirements for the state classes were

- *Usability*: The user should not need to think about how data is stored in different files
- *Flexibility*: We want a general interface to write data in multiple file formats (YAML, HDF, zarr, ...)
- *Stability*: Future versions of the package should be able to read older files even when the internal definitions of file formats change
- *Extensibility*: States should be subclasses to implement more complicated requirements (e.g., particular serialization)
- *Self-explainability*: The files should in principle contain all information to reconstruct the data, even if the `py-modelrunner` package is no longer available.
- *Efficiency*: The files should only store necessary information.

The last point results in particular constraints if we want to store temporal simulation results.
In most cases, there are are some data that are kept fixed for the simulation (describing physical parameters) and others that evolve with time.
We denote by `attributes` the parameters that are kept fixed and by `data` the data that varies over time.
The state classes are already prepared to deal with such data, in conjuction with the :mod:`~modelrunner.state.trajectory` module.

We provide four basic classes that can deal with state data of different type:

- :class:`~modelrunner.state.array.ArrayState` contains a single numpy array
- :class:`~modelrunner.state.array_collection.ArrayCollectionState` contains multiple numpy arrays
- :class:`~modelrunner.state.object.ObjectState` contains a single serializable python object
- :class:`~modelrunner.state.dict.DictState` contains a dictionary of states to allow for nesting

All state classes can be sub-classed to adjust to specialized needs. This will often be
necessary if some attributes cannot be serialized automatically or if the data requires
some modifications before storing. To facilitate control over how data is written and
read, we provide the :attr:`~modelrunner.state.base.StateBase._state_attributes_store`
and :attr:`~modelrunner.state.base.StateBase._state_data_store` attributes which should
return respective attributes and data in a form that can be stored directly. When the
object will be restored during reading, the
:meth:`~modelrunner.state.base.StateBase._state_init` method is used to set the
properties of an object.
