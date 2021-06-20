#!/usr/bin/env bash

# set the likely paths for the pde and droplet package for local testing
export MYPYPATH=../py-pde:../py-droplets:$MYPYPATH

./run_tests.py --types 

    
    