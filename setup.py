import setuptools
from msbench import __version__


def read_requirements():
    reqs = []
    with open('requirements.txt', 'r') as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs


setuptools.setup(
    name="msbench",
    version=__version__,
    author="Model ToolChain",
    author_email="",
    description=("Tool for training or finetuning a sparse model"),
    python_requires='>=3.6',
    packages=setuptools.find_packages(exclude=[".helper", ".helper.*", "helper.*", "helper"]),  # noqa E501
    classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux"),
    install_requires=read_requirements()
)
