#!/usr/bin/env python3


from distutils.core import setup


setup(name='audio2tones',
      python_requires='>=3.8',
      version='1.0',
      description='',
      author='',
      author_email='',
      scripts=['audio2tones.py'],
      install_requires=[
          "audiofile",
          "numpy",
          "scipy",
          "tqdm"
      ],
      license="AGPL"
      )
