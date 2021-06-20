"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""
import argparse

from .job import submit_job


def main():
    """submit a script using command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("script", help="The script that should be run")

    parser.add_argument("-n", "--name", default="job", help="Name of model run")

    parser.add_argument(
        "-p",
        "--parameters",
        metavar="JSON",
        help="Dictionary of parameters for the model",
    )

    parser.add_argument("-o", "--output", required=True, help="Path to output file")

    parser.add_argument(
        "-t",
        "--template",
        default="qsub",
        help="Name of the template for job submission",
    )

    args = parser.parse_args()

    stdout, stderr = submit_job(
        args.script,
        output=args.output,
        name=args.name,
        parameters=args.parameters,
        template=args.template,
    )
    print(stdout, stderr)


if __name__ == "__main__":
    main()
