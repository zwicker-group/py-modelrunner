from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup

BASE_PATH = Path(__file__).resolve().parent


# read the version from the particular file
with open(BASE_PATH / "modelrunner" / "version.py", "r") as f:
    exec(f.read())

DOWNLOAD_URL = (
    f"https://github.com/zwicker-group/py-modelrunner/archive/v{__version__}.tar.gz"
)


# read the requirements from requirements.txt
try:
    with open(BASE_PATH / "requirements.txt", "r") as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]
except FileNotFoundError:
    # fall-back for conda, where requirements.txt apparently does not work
    print("Cannot find requirements.txt")
    install_requires = [
        "jinja2>=3",
        "h5py>=3.5",
        "numpy>=1.18.0",
        "pandas>=1.2",
        "PyYAML>=5",
        "tqdm>=4.45",
    ]


# read the description from the README file
with open(BASE_PATH / "README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="py-modelrunner",
    packages=find_packages(),
    zip_safe=False,  # this is required for mypy to find the py.typed file
    version=__version__,
    license="MIT",
    description="Python classes for organizing (HPC) simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="David Zwicker",
    author_email="david.zwicker@ds.mpg.de",
    url="https://github.com/zwicker-group/py-modelrunner",
    download_url=DOWNLOAD_URL,
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
