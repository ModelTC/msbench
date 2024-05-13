#!/bin/bash

# cmd: sh run_dist.sh 8 ToolChain

msb=/path to sparsity file folder

jobname=msb

export PYTHONPATH=$msb:$PYTHONPATH

g=$(($1<8?$1:8))
spring.submit run -n$1 \
                  --ntasks-per-node=$g \
                  -p $2 \
                  --gpu \
                  --cpus-per-task=2 \
                  --job-name=$jobname \
                  --quotatype=auto \
"nohup python pts.py --backend linklink \
                --config configs/pst_res18_pr_50_layer_2w_1e-5_1k.yaml \
                > msb_log.txt 2>&1 &"
