# setup.py

from setuptools import setup, find_packages

setup(
    name='soo_functions',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'torch',
    ],
    description='A collection of utility functions by Soo',
    author='Sooyoung Her',
    author_email='sooyoung.wind@gmail.com',
    url='https://github.com/sooyoung-wind/soo_functions',
)
