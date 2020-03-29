from setuptools import setup

import simplerl

setup(
    name = 'simplerl',
    version = simplerl.__version__,
    author = 'jeremy',
    author_email = 'jrmsayag@gmail.com',
    maintainer = 'jeremy',
    maintainer_email = 'jrmsayag@gmail.com',
    packages = [
        'simplerl',
        'simplerl.optim',
        'simplerl.qfunction'
    ],
    python_requires='>=3',
    install_requires = [
        'numpy'
    ],
    description = 'A very basic (deep) reinforcement learning implementation.',
    platforms = 'ALL',
)
