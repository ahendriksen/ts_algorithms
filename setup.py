#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import os.path

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

with open(os.path.join('ts_algorithms','VERSION')) as version_file:
    version = version_file.read().strip()

requirements = [
    # Add your project's requirements here, e.g.,
    # 'astra-toolbox',
    # 'sacred>=0.7.2',
    # 'tables==3.4.4',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'recommonmark',
    # Other
    'watchdog',
    'coverage',
    'pytest',
    'pytest-runner',
    ]

setup(
    author="Allard Hendriksen",
    author_email='allard.hendriksen@cwi.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="A collection of common algorithms for tomosipo",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ts_algorithms',
    name='ts_algorithms',
    packages=find_packages(include=['ts_algorithms']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={ 'dev': dev_requirements },
    url='https://github.com/ahendriksen/ts_algorithms',
    version=version,
    zip_safe=False,
)
