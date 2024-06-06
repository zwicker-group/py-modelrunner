#!/bin/bash

export PYTHONPATH=../py-modelrunner  # likely path of package, relative to current base path

if [ ! -z $1 ] 
then 
    # test pattern was specified 
    echo 'Run unittests with pattern '$1':'
    ./run_tests.py --unit --pattern "$1" -- \
        -o log_cli=true --log-cli-level=debug -vv
else
    # test pattern was not specified
    echo 'Run all unittests:'
    ./run_tests.py --unit -- \
        -o log_cli=true --log-cli-level=debug -vv
fi
