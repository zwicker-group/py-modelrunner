#!/usr/bin/env bash
#
# This script formats the code of this package
#

# format imports
echo "Formating import statements..."
isort ..

# format rest of the code
for dir in model examples ; do
    echo "Formating files in ${dir}..."
    black -t py37 ../${dir}
done