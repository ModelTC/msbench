#!/bin/bash

# cmd: sh run.sh

msb=/path to sparsity file folder

export PYTHONPATH=$msb:$PYTHONPATH

nohup python pts.py --launch=pytorch --ng 1 --nm 1 \
                --config configs/pst_res18_pr_50_layer_2w_1e-5_1k.yaml \
                > msb_log.txt 2>&1 &
