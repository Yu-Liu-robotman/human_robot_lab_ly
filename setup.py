from distutils.core import setup

from setuptools import find_packages

setup(
    name="humanoid",
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
        "IsaacLab",
        "rsl-rl-lib>=2.3.0",
    ],
)
