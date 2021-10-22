from setuptools import setup, find_packages
from distutils.core import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='thex_model',
      packages=find_packages(),
      version='4.0',
      description='THEx Model: Package for classifying astronomical transients using host galaxies for THEx collaboration.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Marina Kisley',
      author_email='marinaki@email.arizona.edu',
      python_requires='>=3.6'
      )
