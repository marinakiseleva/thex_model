from setuptools import setup, find_packages
from distutils.core import setup

setup(name='thex_model',
      # packages=find_packages(),
      packages=['thex_data', 'models', 'mainmodel', 'classifiers', 'utilities'],
      version='2.5',
      description='THEx Model',
      author='Marina Kisley',
      author_email='marinaki@email.arizona.edu'
      )
