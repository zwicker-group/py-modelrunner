Main structure of the package
=============================

The following graph summarizes the classes responsible for defining models and
corresponding states.

.. graphviz::

   digraph Structure {
      node [shape=box]  /* default attributes */

      State [label=<<b>State</b><br/>Defines simulation state:<br align="left"/>- data ... degrees of freedom<br align="left"/>- attributes ... additional information>, href="../packages/modelrunner.html#modelrunner.state.StateBase", target="_top"];
      Trajectory [label=<<b>Trajectory</b><br/>States as a function of time>, href="../packages/modelrunner.html#modelrunner.state.Trajectory", target="_top"];
      Model [label=<<b>Model</b><br/>Describing simulation<br align="left"/>- Contains model parameters>, href="../packages/modelrunner.html#module-modelrunner.model", target="_top"];
      Result [label=<<b>Result</b><br/>Combination of model and a state<br align="left"/>Contains all relevant information for further analysis>, href="../packages/modelrunner.html#modelrunner.results.Result", target="_top"];
      ResultCollection [label=<<b>ResultCollection</b><br/>Results from different model parameters<br align="left"/>- Deals with parameter sweeps<br align="left"/>>, href="../packages/modelrunner.html#modelrunner.results.ResultCollection", target="_top"];

      State -> Result;
      State -> Trajectory;
      Model -> Result;
      Result -> ResultCollection;
   }