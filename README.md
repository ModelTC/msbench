<div align="center">
  <img src="resources/logo.png" width="600"/>

</div>

------------

## Introduction

MSBench is an open-source model sparsification toolkit based on PyTorch fx.


## Quick start pts example without installation

```
git clone git@gitlab.bj.sensetime.com:spring2/sparsity.git
cd sparsity/applications/imagenet_example/PTS
# modify configs/pst_res18_pr_50_layer_2w_1e-5_1k.yaml

# for dist train
# modify msb path in run_dist.sh
sh run_dict.sh 8 ToolChain

# for one gpu on your owm machine
# modify msb path in run.sh
sh run.sh
```

## TO DO LIST (PTS部分)
1. eval支持多卡


## Installation
Clone the project from gitlab and install it
```
git clone git@gitlab.bj.sensetime.com:spring2/sparsity.git
cd sparsity
python setup.py install
```
or directly through spring Pypi for online installation
```
pip install http://10.10.40.93/packages/msbench-0.0.1-py3-none-any.whl
```

## Documentation

MSBench aims to support (1) various deployable sparsification algorithms and (2) hardware backend libraries to facilitate the development of the community.

For the detailed information, please refer to [msbench documentation](http://spring.sensetime.com/docs/sparsity/index.html)


## How to Create a Wheel file

First, you will need to install a couple of Python packages.

```
pip install wheel setuptools
```

Then, type the following command to create the wheel file

```
python setup.py sdist bdist_wheel || true
```

The file will be generated in **dist** folder
## License

This project is released under the [Apache 2.0 license](LICENSE).