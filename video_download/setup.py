import os
from distutils.core import setup

with open('requirements.txt') as f:
        required = f.read().splitlines()

setup(name='videodownloader',
        version='1.0',
        author='dinu thomas',
        author_email='dinu.thomas@gmail.com',
        url='http://example.com',
        py_modules=['VideoDownloader'],
        install_requires=[required,]
        )
