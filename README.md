# py-job
Python classes for handling and running physical simulations


Installation
============
The package can simply be cloned from github. The only compulsory package is `numpy`, but
additionally packages might be necessary for the full functionality (e.g., `jinja2` and
`pandas`).


Usage
=====

This package has multiple purposes that are described in more detail below. Additional
examples can be found in the `examples` folder.

Creating models
---------------
The package introduces a base class `ModelBase` that describes the bare structure all
models are supposed to have.
Custom models can be created by inheriting from `ModelBase` and defining suitable
parameters:

```python
from job import ModelBase

class MyModel(ModelBase):  # define custom model

    # defines supported parameters with default values
    parameters_default = {"a": 1, "b": 2}

    def __call__(self):
        """calculate the actual model"""
        return self.parameters["a"] * self.parameters["b"]
        
        
model = MyModel({"a" : 3})
```

The last line actually creates an instance of the model with custom parameters.

Alternatively, a model can also be defined from a simple function:

```python
from job import get_function_model

def multiply(a=1, b=2):
    return a * b

model = get_function_model(multiply, {"a": 3})
```

The main aim of defining models like this is to provide a unified interface for
running models for the subsequent sections.


Run models from command line
----------------------------

Models can be run with different parameters. In both examples shown above, the model
can be run from within the python code by simply calling the model instance: `model()`.
In the cases shown above, these calls will simply return `6`.

In typical numerical simulations, models need to be evaluated for many different
parameters. The packages facilitates this by providing a special interface to set
arguments from the command line. To show this, either one of the model definitions
given above can be saved as a python file `model.py`. Using the special call
`python -m job model.py` provides a command line interface for adjusting model parameters.
In particular, calling `python -m job model.py -h` displays all possible settings.


Submit models to an HPC queue
-----------------------------
The package also provides methods to submit scripts to an high performance compute (HPC)
system. 
A simple full script displaying this reads

```python
from job import make_model, submit_job

@make_model
def multiply(a=1, b=2):
    return a * b

if __name__ == "__main__":
    submit_job(__file__, parameters={'a': 2}, output="data.hdf5", method="local")
```
Here, the `output` argument specifies a file to which the results are written, while
`method` chooses how the script is submitted.

In particular, this method allows submitting the same script with multiple different
parameters to conduct a parameter study:

```python
from job import make_model, submit_job

@make_model
def multiply(a=1, b=2):
    return a * b

if __name__ == "__main__":
    for a in range(5):
        submit_job(__file__, parameters={'a': a}, output=f"data_{a}.hdf5", method="local")
```

Note that the safe-guard `if __name__ == "__main__"` is absolutely crucial to ensure that
jobs are only submitted during the initial run and not when the file is imported again
when the actual jobs start. It is also important to choose unique file names for the
`output` flag since otherwise different jobs overwrite each others data.


Collating results
-----------------
Finally, the package also provides some rudimentary support for collection results from
many different simulations that have been run in parallel. In particular, the class
`ResultCollection` provides a class method `from_folder` to scan a folder for result files.
For instance, the data from the multiple jobs ran above can be collected using

```python
from job import ResultCollection

results = ResultCollection.from_folder(".", pattern="data_*.hdf5")
print(results.dataframe)
```

This example should print all results using a pandas Dataframe, where each row
corresponds to a separate simulation.


Development
===========
The package is in an early phase and breaking changes are thus likely.