from setuptools import setup, find_packages
from EnsembleTrees import __version__

setup(name='EnsembleTrees',
    version=__version__,
    description='EnsembleTrees - a tree-based algorithm for extrapolation',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Topic :: Cheminformatics :: Extrapolation',
    ],
    url='https://github.com/funatsu-lab/Ensemble-trees',
    author='Kohei Numata, Kenichi Tanaka',
    author_email='knumata@chemsys.t.u-tokyo.ac.jp',
    packages=find_packages(),
    install_requires=['numpy>=1.18'],
    )
