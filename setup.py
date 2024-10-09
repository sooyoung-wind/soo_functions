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
        # 필요한 다른 패키지들을 여기에 추가하세요
    ],
    description='A collection of utility functions by Soo',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your_username/soo_functions_project',
)
