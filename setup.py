#!/usr/bin/env python
try:
  from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt', 'r') as f:
	r = f.readlines()

setup(name='pdemo',
      version='0.0.1',
      description='PDF Demo',
      author='Kyle Bogdan',
      author_email='kyle.bogdan@slalom.com',
      url='https://github.com/mathematicalmichael/pdemo/',
      packages=['pdemo'],
      install_requires=r,
)
