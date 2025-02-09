from setuptools import setup, find_packages

setup(
    name="autograd",
    version="0.1",
    packages=find_packages(include=['autograd', 'autograd.*', 'tests']),
    install_requires=[
        'numpy',
        'rich',
        'torch'
    ],
)