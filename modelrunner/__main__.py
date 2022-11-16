"""
Main module allowing to use the package to wrap existing code

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import sys

from modelrunner import Result, run_script


def run_script_from_command_line() -> Result:
    """helper function that runs a model from flags specified at the command line"""
    # get the script name from the command line
    try:
        script_path = sys.argv[1]
    except IndexError:
        print("Require job script as first argument", file=sys.stderr)
        sys.exit(1)

    return run_script(script_path, sys.argv[2:])


if __name__ == "__main__":
    run_script_from_command_line()
