from setuptools import setup, find_packages
from StochasticThresholdModelTrees import __version__

setup(name='Stochastic Threshold Model Trees',
    version=__version__,
    description='Stochastic Threshold Model Trees - a tree-based algorithm for extrapolation',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Cheminformatics :: Extrapolation',
    ],
    url='https://github.com/funatsu-lab/Stochastic-Threshold-Model-Trees',
    author='Kohei Numata',
    author_email='knumata@chemsys.t.u-tokyo.ac.jp',
    packages=find_packages(),
    install_requires=['sklearn==0.21', 'numpy>=1.17', 'joblib==0.13', 'pandas>=0.25', 'matplotlib>=3.1', 'seaborn>=0.9'],
    )
