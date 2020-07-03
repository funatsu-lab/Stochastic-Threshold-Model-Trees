from setuptools import setup, find_packages
from EnsembleTrees import __version__

setup(name='EnsembleTrees',
    version=__version__,
    description='Stochastic Threshold Model Tree - a tree-based algorithm for extrapolation',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Topic :: Cheminformatics :: Extrapolation',
    ],
    url='https://github.com/funatsu-lab/Stochastic-Threshold-Model-Tree',
    author='Kohei Numata, Kenichi Tanaka',
    author_email='knumata@chemsys.t.u-tokyo.ac.jp',
    packages=find_packages(),
    install_requires=['numpy>=1.17', 'joblib>=0.13'],
    )
