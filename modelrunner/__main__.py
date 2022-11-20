"""
Main module allowing to use the package to wrap existing code

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import sys

from modelrunner import Result, run_script


def run_script_from_command_line() -> Result:
    """helper function that runs a model from flags specified at the command line

    The function detects models automatically by trying several methods until one yields
    a unique model to run:

    * A model that have been marked as default by :func:`set_default`
    * A function named `main`
    * A model instance if there is exactly one (throw error if there are many)
    * A model class if there is exactly one (throw error if there are many)
    * A function if there is exactly one (throw error if there are many)

    Returns:
        :class:`Result`: The result of running the model
    """
    # get the script name from the command line
    try:
        script_path = sys.argv[1]
    except IndexError:
        print("Require job script as first argument", file=sys.stderr)
        sys.exit(1)

    return run_script(script_path, sys.argv[2:])


if __name__ == "__main__":
    run_script_from_command_line()
