from setuptools import setup

import simplenn

setup(
    name = 'simplenn',
    version = simplenn.__version__,
    author = 'jeremy',
    author_email = 'jrmsayag@gmail.com',
    maintainer = 'jeremy',
    maintainer_email = 'jrmsayag@gmail.com',
    packages = [
        'simplenn',
        'simplenn.activations',
        'simplenn.loss',
        'simplenn.loss.simulation',
        'simplenn.network',
        'simplenn.optim'
    ],
    python_requires='>=3',
    install_requires = [
        'numpy'
    ],
    description = 'A very basic (deep) neural networks implementation.',
    platforms = 'ALL',
)
