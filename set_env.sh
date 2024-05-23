#!/usr/bin/env bash
export ROMA_HOME="$(pwd)"
export NN_HOME="$ROMA_HOME/nn"
export LOG_DIR="$NN_HOME/logs"
export PYTHONPATH="$ROMA_HOME:$PYTHONPATH"
export PYTHONPATH="$NN_HOME:$PYTHONPATH"
export DATAPATH="$ROMA_HOME/data"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
#source activate hgcn  # replace with source hgcn/bin/activate if you used a virtualenv
