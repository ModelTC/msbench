#!/bin/bash

# cmd: sh run_dist.sh 8 ToolChain

msb=/path to msbench file folder

jobname=msb

export PYTHONPATH=$msb:$PYTHONPATH

python pts.py --config configs/pst_res18_pr_50_layer_2w_1e-5_1k.yaml 
