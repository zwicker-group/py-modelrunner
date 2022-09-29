"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""
import argparse

from .job import submit_job


def main():
    """submit a script using command line arguments"""
    parser = argparse.ArgumentParser(
        prog="python -m modelrunner.run", description="Run a script as a job"
    )

    parser.add_argument("script", help="The script that should be run")

    parser.add_argument("-n", "--name", default="job", help="Name of job")

    parser.add_argument(
        "-p",
        "--parameters",
        metavar="JSON",
        help="JSON-encoded dictionary of parameters for the model",
    )

    parser.add_argument("-o", "--output", help="Path to output file", metavar="PATH")
    parser.add_argument(
        "--overwrite",
        nargs="?",
        default="error",
        const="overwrite",
        choices=["error", "warn_skip", "silent_skip", "overwrite", "silent_overwrite"],
        help="Decide how existing data should be handled",
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
        overwrite_strategy=args.overwrite,
    )
    print(stdout, stderr)


if __name__ == "__main__":
    main()
