# py-modelrunner


[![Build status](https://github.com/zwicker-group/py-modelrunner/workflows/build/badge.svg)](https://github.com/zwicker-group/py-modelrunner/actions?query=workflow%3Abuild)
[![Documentation Status](https://readthedocs.org/projects/py-modelrunner/badge/?version=latest)](https://py-modelrunner.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/py-modelrunner.svg)](https://badge.fury.io/py/py-modelrunner)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/py-modelrunner.svg)](https://anaconda.org/conda-forge/py-modelrunner)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/zwicker-group/py-modelrunner/branch/main/graph/badge.svg)](https://codecov.io/gh/zwicker-group/py-modelrunner)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



This package provides python classes for handling and running physical simulations.
The main aim is to easily wrap simulation code and deal with input and output automatically.
The package also facilitates submitting simulations to high performance computing
environments and it provides functions for running parameter scans.


Installation
============
The package can simply be cloned from github, but it is also available on `pip` and `conda`:

```bash
pip install py-modelrunner
```

Usage
=====

This package has multiple purposes that are described in more detail below. Additional
examples can be found in the `examples` folder.

Minimal example
---------------
Assume you have written a python simulation in form of a simple script that defines a
function with several arguments, like so

```python
def main(a: float = 1, b: int = 2, negate: bool = False):
    res = a ** b
    if negate:
    	res *= -1
    return res
```

The `modelrunner` package now allows you to wrap a convenient command line interface
around this simple function. Assuming the script is saved in a file called `script.py`,
calling `python -m modelrunner script.py -h` shows the follwing help

```
usage: test.py [-h] [--a VALUE] [--b VALUE] [--negate] [--json JSON] [-o PATH]

optional arguments:
  -h, --help            show this help message and exit
  --json JSON           JSON-encoded parameter values. Overwrites other parameters. (default: None)
  -o PATH, --output PATH
                        Path to output file. If omitted, no output file is created. (default: None)

  --a VALUE             Parameter `a` (default: 1)
  --b VALUE             Parameter `b` (default: 2)
  --negate              Parameter `negate` (default: False)
```

Consequently, the function can be called using `python -m modelrunner script.py --a 2 --b 3 --negate -o result.yaml`,
which produces a file `result.yaml` with the following content:

```yaml
info:
  time: # TIMESTAMP
model:
  class: main
  description: null
  name: main
  parameters:
    a: 2.0
    b: 3
    negate: true
result: -8.0
```

This file not only contains the result, but also metainformation including the
parameters to run the simulation and the time when it was started.


Creating models
---------------
The package introduces a base class `ModelBase` that describes the bare structure all
models are supposed to have.
Custom models can be created by inheriting from `ModelBase` and defining suitable
parameters:

```python
from modelrunner import ModelBase

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
from modelrunner import make_model

@make_model
def multiply(a=1, b=2):
    return a * b

model = multiply(a=3)
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
`python -m modelrunner model.py` provides a command line interface for adjusting model parameters.
The supported parameters can be obtained with the following command

```console
$ python -m modelrunner model.py --help

usage: model.py [-h] [--a VALUE] [--b VALUE] [-o PATH] [--json JSON]

optional arguments:
  -h, --help            show this help message and exit
  -o PATH, --output PATH
                        Path to output file. If omitted, no output file is created. (default: None)
  --json JSON           JSON-encoded parameter values. Overwrites other parameters. (default: None)

  --a VALUE             Parameter `a` (default: 1)
  --b VALUE             Parameter `b` (default: 2)
```

This can be helpful to call a model automatically and save the result. For instance, by
calling `python -m modelrunner model.py -h --a 3 -o result.yaml`, we obtain a file `result.yaml` that
looks something like this:

```yaml
model:
  class: multiply
  name: multiply
  parameters:
    a: 3
    b: 2
result: 6
```

Other supported output formats include JSON (extension `.json`) and HDF (extension `.hdf`).


Submit models to an HPC queue
-----------------------------
The package also provides methods to submit scripts to an high performance compute (HPC)
system. 
A simple full script displaying this reads

```python
from modelrunner import make_model, submit_job

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
from modelrunner import make_model, submit_job

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

We also support submitting multiple jobs of a parameter study:

```python
from modelrunner import make_model, submit_jobs

@make_model
def multiply(a=1, b=2):
    return a * b

if __name__ == "__main__":
    submit_jobs(__file__, parameters={'a': range(5)}, output_folder="data", method="local")
```

Finally, the packages also offers a method to submit a model script to the cluster using
a simple command: `python3 -m modelrunner.run script.py`. This command also offers multiple options
that can be adjusted using command line arguments:

```
usage: python -m modelrunner.run [-h] [-n NAME] [-p JSON] [-o PATH] [-f] [-m METHOD] [-t PATH] script

Run a script as a job

positional arguments:
  script                The script that should be run

optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  Name of job
  -p JSON, --parameters JSON
                        JSON-encoded dictionary of parameters for the model
  -o PATH, --output PATH
                        Path to output file
  -f, --force           Overwrite data if it already exists
  -m METHOD, --method METHOD
                        Method for job submission
  -t PATH, --template PATH
                        Path to template file for submission script
```

Collating results
-----------------
Finally, the package also provides some rudimentary support for collection results from
many different simulations that have been run in parallel. In particular, the class
`ResultCollection` provides a class method `from_folder` to scan a folder for result files.
For instance, the data from the multiple jobs ran above can be collected using

```python
from modelrunner import ResultCollection

results = ResultCollection.from_folder(".", pattern="data_*.hdf5")
print(results.dataframe)
```

This example should print all results using a pandas Dataframe, where each row
corresponds to a separate simulation.


Development
===========
The package is in an early phase and breaking changes are thus likely.
