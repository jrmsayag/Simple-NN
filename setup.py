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
        'simplenn.evaluation',
        'simplenn.evaluation.loss',
        'simplenn.evaluation.simulation',
        'simplenn.optim',
        'simplenn.optim.evolution',
        'simplenn.optim.gradient',
        'simplenn.optim.qlearning',
        'simplenn.structures',
        'simplenn.structures.network',
        'simplenn.structures.network.activations',
        'simplenn.structures.qfunction'
    ],
    python_requires='>=3',
    install_requires = [
        'numpy'
    ],
    extras_require = {
        'examples':  ['gym', 'wandb']
    },
    description = 'A very basic implementation of various machine-learning aglorithms.',
    platforms = 'ALL',
)
