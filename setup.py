import os
import torch
import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="DGraph",
    py_modules=["DGraph"],
    install_requires=[
        "torch",
        "numpy",
    ],
)
