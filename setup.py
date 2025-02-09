from setuptools import setup, find_packages

setup(
    name="chibigrad",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "tests": ["torch>=2.0.0"]
    }
)