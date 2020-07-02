from setuptools import setup, find_packages

setup(name='EnsembleTrees',
    version=__version__,
    description='Fragment2vec - an appropriate algorithm for substructure representation in chemistry',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Topic :: Cheminformatics :: Featurization',
    ],
    url='https://github.com/funatsu-lab/frag2vec',
    author='Shojiro Shibayama',
    author_email='shojiro.shibayama@gmail.com',
    license='BSD 3-clause',
    packages=find_packages(),
    install_requires=['numpy>=1.18', 'tqdm>=4.42', 'pytorch>=1.4',
                     'torchvision>=0.5', 'rdkit>=2019.09', 'networkx>=2.4'],
    )
