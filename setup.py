#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="uffm",
    version="0.1.0",
    description="Uncertainty Flow Foundation Model - Distilling pretrained LLMs into flow-based models",
    author="pqhung",
    author_email="",
    url="https://github.com/user/uffm",
    install_requires=[
        "lightning",
        "hydra-core",
        "transformers",
        "datasets",
        "evaluate",
        "accelerate",
        "peft",
        "transformer-lens",
        "torchdiffeq",
        "flow-matching",
    ],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
