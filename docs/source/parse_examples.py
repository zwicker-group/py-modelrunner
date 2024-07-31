#!/usr/bin/env python3

import pathlib
import re

# Root direcotry of the package
ROOT = pathlib.Path(__file__).absolute().parents[2]
# directory where all the examples reside
INPUT = ROOT / "examples"
# directory to which the documents are writen
OUTPUT = ROOT / "docs" / "source" / "examples"


def main():
    """Parse all examples and write them in a special example module."""
    # create the output directory
    OUTPUT.mkdir(parents=True, exist_ok=True)

    regex = r'"""((?!""").*?)"""'

    # iterate over all examples
    for path_in in INPUT.glob("*.py"):
        path_out = OUTPUT / (path_in.stem + ".rst")
        print(f"Found example {path_in}")
        with path_in.open("r") as file_in, path_out.open("w") as file_out:
            # write the header for the rst file
            file_out.write(path_in.name + "\n")
            file_out.write("-" * len(path_in.name) + "\n\n")

            # read all content
            file_content = file_in.read()

            # identify shebang
            if file_content.startswith("#"):
                shebang, file_content = file_content.split("\n", 1)
                if shebang == "#!/usr/bin/env python3":
                    shebang = ""  # filter trivial case
            else:
                shebang = None

            # locate file docstring
            matches = re.search(regex, file_content, re.DOTALL)
            if matches:
                for line in matches.group(1).split("\n"):
                    if ".. codeauthor::" in line:
                        continue
                    file_out.write(f"{line}\n")

                file_content = file_content[matches.end(0) :]

            # write actual code
            file_out.write(".. code-block:: python\n\n")
            if shebang:
                file_out.write(f"    {shebang}\n\n")
            header = True
            for line in file_content.split("\n"):
                # skip the shebang, comments and empty lines in the beginning
                if header and (line.startswith("#") or len(line.strip()) == 0):
                    continue
                header = False  # first real line was reached
                file_out.write(f"    {line}\n")


if __name__ == "__main__":
    main()
