"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import errno
import itertools
import json
import logging
import os
import pipes
import subprocess as sp
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

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


def get_job_name(
    base: str, args: Optional[Dict[str, Any]] = None, length: int = 7
) -> str:
    """create a suitable job name

    Args:
        base (str): The stem of the job name
        args (dict): Parameters to include in the job name
        length (int): Length of the abbreviated parameter name

    Returns:
        str: A suitable job name
    """
    if args is None:
        args = {}

    res = base[:-1] if base.endswith("_") else base
    for name, value in args.items():
        if hasattr(value, "__iter__"):
            value_str = "_".join(f"{v:g}" for v in value)
        else:
            value_str = f"{value:g}"
        res += f"_{name.replace('_', '')[:length].upper()}_{value_str}"
    return res


def submit_job(
    script: Union[str, Path],
    output: Union[str, Path, None] = None,
    name: str = "job",
    parameters: Union[str, Dict[str, Any], None] = None,
    *,
    log_folder: Union[str, Path] = "logs",
    method: str = "qsub",
    use_modelrunner: bool = True,
    template: Union[str, Path, None] = None,
    overwrite_strategy: str = "error",
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
            Specifies the submission method. Currently `background`, `foreground`, and
            `qsub` are supported.
        use_modelrunner (bool):
            If True, `script` is envoked with the modelrunner library, e.g. by calling
            `python -m modelrunner {script}`.
        template (str of :class:`~pathlib.Path`):
            Jinja template file for submission script. If omitted, a standard template
            is chosen based on the submission method.
        overwrite_strategy (str):
            Determines what to do when files already exist. Possible options include
            `error`, `warn_skip`, `silent_skip`, `overwrite`, and `silent_overwrite`.
        **kwargs:
            Extra arguments are forwarded as template variables to the script

    Returns:
        tuple: The result `(stdout, stderr)` of the submission call
    """
    from jinja2 import Template

    logger = logging.getLogger("modelrunner.submit_job")

    if template is None:
        template_path = Path(__file__).parent / "templates" / (method + ".template")
    else:
        template_path = Path(template)
    logger.info("Load template `%s`", template_path)
    with open(template_path, "r") as fp:
        script_template = fp.read()

    # prepare submission script
    ensure_directory_exists(log_folder)

    script_args = {
        "LOG_FOLDER": log_folder,
        "JOB_NAME": name,
        "MODEL_FILE": escape_string(script),
        "USE_MODELRUNNER": use_modelrunner,
    }
    for k, v in kwargs.items():
        script_args[k.upper()] = v

    # add the parameters to the job arguments
    job_args = []
    if parameters is not None and len(parameters) > 0:
        if isinstance(parameters, dict):
            parameters = json.dumps(parameters)
        elif not isinstance(parameters, str):
            raise TypeError("Parameters need to be given as a string or a dict")
        job_args.append(f"--json {escape_string(parameters)}")

    logger.debug("Job arguments: `%s`", job_args)

    # add the output folder to the job arguments
    if output:
        output = Path(output)
        if output.is_file():
            # output is an existing file, so we need to decide what to do with this
            if overwrite_strategy == "error":
                raise RuntimeError(f"Output file `{output}` already exists")
            elif overwrite_strategy == "warn_skip":
                warnings.warn(f"Output file `{output}` already exists")
                return "", f"Output file `{output}` already exists"  # do nothing
            elif overwrite_strategy == "silent_skip":
                return "", f"Output file `{output}` already exists"  # do nothing
            elif overwrite_strategy == "overwrite":
                warnings.warn(f"Output file `{output}` will be overwritten")
            elif overwrite_strategy == "silent_overwrite":
                pass
            else:
                raise NotImplementedError(f"Unknown strategy `{overwrite_strategy}`")

        # check whether output points to a directory or whether this should be a file
        if output.is_dir():
            script_args["OUTPUT_FOLDER"] = pipes.quote(str(output))
        else:
            script_args["OUTPUT_FOLDER"] = pipes.quote(str(output.parent))
            job_args.append(f"--output {escape_string(output)}")

    else:
        # if `output` is not specified, save data to current directory
        script_args["OUTPUT_FOLDER"] = "."
    script_args["JOB_ARGS"] = " ".join(job_args)

    # replace parameters in submission script template
    script_content = Template(script_template).render(script_args)
    logger.debug("Script: `%s`", script_content)

    if method == "qsub":
        # submit job to queue
        proc = sp.Popen(
            ["qsub"],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
        )

    elif method in {"background", "foreground"}:
        # run job locally
        proc = sp.Popen(
            ["bash"],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
        )

    else:
        raise ValueError(f"Unknown submit method `{method}`")

    return proc.communicate(script_content)


def submit_jobs(
    script: Union[str, Path],
    output_folder: Union[str, Path],
    name_base: str = "job",
    parameters: Union[str, Dict[str, Any], None] = None,
    *,
    list_params: Optional[Iterable[str]] = None,
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
            that are iterable and not strings and not part of `keep_list` are submitted
            as separate jobs.
        list_params (list):
            List of parameters that are meant to be lists. They will be submitted as
            individual parameters and not iterated over to produce multiple jobs.
        **kwargs:
            All additional parameters are forwarded to :func:`submit_job`.
    """
    if parameters is None:
        parameter_dict = {}
    elif isinstance(parameters, str):
        parameter_dict = json.loads(parameters)
    else:
        parameter_dict = parameters
    if list_params is None:
        list_params = set()

    # detect varying parameters
    params, p_vary = {}, {}
    for name, value in parameter_dict.items():
        if (
            hasattr(value, "__iter__")
            and not isinstance(value, str)
            and name not in list_params
        ):
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
