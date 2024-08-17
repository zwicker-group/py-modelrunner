#!/usr/bin/env python3

import glob
import logging
import os
import subprocess as sp
from pathlib import Path

logging.basicConfig(level=logging.INFO)

OUTPUT_PATH = "packages"


def replace_in_file(infile: Path, replacements, outfile=None):
    """Reads in a file, replaces the given data using python formatting and writes back
    the result to a file.

    Args:
        infile (str):
            File to be read
        replacements (dict):
            The replacements old => new in a dictionary format {old: new}
        outfile (str):
            Output file to which the data is written. If it is omitted, the
            input file will be overwritten instead
    """
    if outfile is None:
        outfile = infile

    with infile.open() as fp:
        content = fp.read()

    for key, value in replacements.items():
        content = content.replace(key, value)

    with outfile.open("w") as fp:
        fp.write(content)


def main(package="modelrunner"):
    # remove old files
    for path in Path(OUTPUT_PATH).glob("*.rst"):
        logging.info("Remove file `%s`", path)
        path.unlink()

    # run sphinx-apidoc
    sp.check_call(
        [
            "sphinx-apidoc",
            "--separate",
            "--maxdepth",
            "4",
            "--output-dir",
            OUTPUT_PATH,
            "--module-first",
            f"../../{package}",  # path of the package
            f"../../{package}/tests",  # ignored path
            f"../../{package}/**/tests",  # ignored path
            f"../../{package}/conftest.py",  # ignore conftest
        ]
    )

    REPLACEMENTS = {
        "Submodules\n----------\n\n": "",
        "Subpackages\n-----------": "**Subpackages:**",
        "sim package\n===========": "Reference manual\n================",
    }

    # replace unwanted information
    for path in Path(OUTPUT_PATH).glob("*.rst"):
        logging.info("Patch file `%s`", path)
        replace_in_file(path, REPLACEMENTS)


if __name__ == "__main__":
    main()
