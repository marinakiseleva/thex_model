from setuptools import setup, find_packages
from distutils.core import setup

setup(name='thex',
      # packages=find_packages('thex_model'),
      packages=['thex_data', 'models'],
      version='2.3',
      description='THEx Model',
      author='Marina Kiseleva',
      author_email='marinaki@email.arizona.edu'
      )
