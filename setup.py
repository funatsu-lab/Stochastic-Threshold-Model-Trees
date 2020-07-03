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
    author='Kohei Numata, Kenichi Tanaka',
    author_email='knumata@chemsys.t.u-tokyo.ac.jp',
    packages=find_packages(),
    install_requires=['numpy', 'joblib', 'pandas', 'matplotlib', 'seaborn'],
    )
