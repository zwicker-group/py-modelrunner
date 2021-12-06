"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import errno
import itertools
import json
import os
import pipes
import subprocess as sp
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from tqdm.auto import tqdm


def escape_string(obj) -> str:
    """escape a string for the command line"""
    return pipes.quote(str(obj))


def ensure_directory_exists(folder):
    """creates a folder if it not already exists"""
    if folder == "":
        return
    try:
        os.makedirs(folder)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise


def get_job_name(base: str, args: Dict[str, Any] = None, length: int = 7) -> str:
    """create a suitable job name

    Args:
        base (str): The stem of the job name
        args (dict): Parameters to include in the job name
        length (int)": Length of the abbreviated parameter name

    Returns:
        str: A suitable job name
    """
    if args is None:
        args = {}

    res = base[:-1] if base.endswith("_") else base
    for name, value in args.items():
        res += f"_{name.replace('_', '')[:length].upper()}_{value:g}"
    return res


def submit_job(
    script: Union[str, Path],
    output: Union[str, Path],
    name: str = "job",
    parameters: Union[str, Dict[str, Any]] = None,
    *,
    log_folder: Union[str, Path] = "logs",
    method: str = "qsub",
    template: Union[str, Path] = None,
    overwrite_files: bool = False,
    **kwargs,
) -> Tuple[str, str]:
    """submit a script to the cluster queue

    Args:
        script (str of :class:`~pathlib.Path`):
            Path to the script file, which contains the model
        output (str of :class:`~pathlib.Path`):
            Path to the output file, where all the results are saved
        name (str):
            Name of the job
        parameters (str or dict):
            Parameters for the script, either as a python dictionary or a string
            containing a JSON-encoded dictionary.
        log_folder (str of :class:`~pathlib.Path`):
            Path to the logging folder
        method (str):
            Specifies the submission method. Currently `qsub` and `local` are supported.
        template (str of :class:`~pathlib.Path`):
            Jinja template file for submission script. If omitted, a standard template
            is chosen based on the submission method.
        overwrite_files (bool):
            Determines whether output files are overwritten
        **kwargs:
            Extra arguments are forwarded as template variables to the script

    Returns:
        tuple: The result `(stdout, stderr)` of the submission call
    """
    from jinja2 import Template

    if template is None:
        template_path = Path(__file__).parent / "templates" / (method + ".template")
    else:
        template_path = Path(template)
    with open(template_path, "r") as fp:
        script_template = fp.read()

    # prepare submission script
    ensure_directory_exists(log_folder)

    script_args = {
        "LOG_FOLDER": log_folder,
        "JOB_NAME": name,
        "MODEL_FILE": escape_string(script),
    }
    for k, v in kwargs.items():
        script_args[k.upper()] = v

    job_args = []
    if parameters is not None:
        if isinstance(parameters, dict):
            parameters = json.dumps(parameters)
        elif not isinstance(parameters, str):
            raise TypeError("Parameters need to be given as a string or a dict")
        job_args.append(f"--json {escape_string(parameters)}")

    if output:
        output = Path(output)
        if output.is_file() and not overwrite_files:
            raise RuntimeError(f"Output file `{output}` already exists")
        script_args["OUTPUT_FOLDER"] = pipes.quote(str(output.parent))
        job_args.append(f"--output {escape_string(output)}")
    else:
        script_args["OUTPUT_FOLDER"] = "."
    script_args["JOB_ARGS"] = " ".join(job_args)

    # replace parameters in submission script template
    script = Template(script_template).render(script_args)

    # submit job to queue
    if method == "qsub":
        proc = sp.Popen(
            ["qsub"],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
        )

    elif method == "local":
        proc = sp.Popen(
            ["bash"],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
        )

    else:
        raise ValueError(f"Unknown submit method `{method}`")

    return proc.communicate(script)


def submit_jobs(
    script: Union[str, Path],
    output_folder: Union[str, Path],
    name_base: str = "job",
    parameters: Union[str, Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """submit many jobs of the same script with different parameters to the cluster

    Args:
        script (str of :class:`~pathlib.Path`):
            Path to the script file, which contains the model
        output_folder (str of :class:`~pathlib.Path`):
            Path to the output folder, where all the results are saved
        name_base (str):
            Base name of the job. An automatic name is generated on this basis.
        parameters (str or dict):
            Parameters for the script, either as a python dictionary or a string
            containing a JSON-encoded dictionary. All combinations of parameter values
            that are iterable and not strings are submitted as separate jobs.
        **kwargs:
            All additional parameters are forwarded to :func:`submit_job`.
    """
    if parameters is None:
        parameter_dict = {}
    elif isinstance(parameters, str):
        parameter_dict = json.loads(parameters)
    else:
        parameter_dict = parameters

    # detect varying parameters
    params, p_vary = {}, {}
    for name, value in parameter_dict.items():
        if hasattr(value, "__iter__") and not isinstance(value, str):
            p_vary[name] = value
        else:
            params[name] = value

    # build the list of all varying arguments
    p_vary_list = [
        dict(zip(p_vary.keys(), values))
        for values in itertools.product(*p_vary.values())
    ]

    # submit jobs with all parameter variations
    for p_job in tqdm(p_vary_list):
        params.update(p_job)
        name = get_job_name(name_base, p_job)
        output = Path(output_folder) / f"{name}.hdf5"
        submit_job(script, output=output, name=name, parameters=params, **kwargs)
