"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""
import argparse

from .job import submit_job


def main():
    """submit a script using command line arguments"""
    parser = argparse.ArgumentParser(description="Submit a script to a queue.")

    parser.add_argument("script", help="The script that should be run")

    parser.add_argument("-n", "--name", default="job", help="Name of model run")

    parser.add_argument(
        "-p",
        "--parameters",
        metavar="JSON",
        help="Dictionary of parameters for the model",
    )

    parser.add_argument("-o", "--output", help="Path to output file", metavar="PATH")
    parser.add_argument(
        "-f", "--force", action="store_true", default=False, help="Overwrite data"
    )
    parser.add_argument(
        "-m", "--method", default="qsub", help="Method for job submission"
    )
    parser.add_argument(
        "-t",
        "--template",
        help="Path to template file for submission script",
        metavar="PATH",
    )

    args = parser.parse_args()

    stdout, stderr = submit_job(
        args.script,
        output=args.output,
        name=args.name,
        parameters=args.parameters,
        method=args.method,
        template=args.template,
        overwrite_files=args.force,
    )
    print(stdout, stderr)


if __name__ == "__main__":
    main()
