import sys

from setuptools import setup, find_packages

setup(
    name='ch_pipeline',
    version=0.1,

    packages=find_packages(),
    package_data={ "ch_pipeline": [ "data/*" ] },

    author="CHIME collaboration",
    author_email="richard@phas.ubc.ca",
    description="CHIME Pipeline",
    url="http://bitbucket.org/chime/ch_pipeline/",
)
