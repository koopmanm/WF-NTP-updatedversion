"""
Copyright (C) 2017  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='multiwormtracker',
      version='3.0.0',
      description='Wide-field nematode tracking platform.',
      long_description=readme(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering',
      ],
      keywords='nematode worm tracker',
      url='https://github.com/koopmanm/WF-NTPv2.0',
      author='TODO',
      author_email='TODO',
      license='GPl v3',
      packages=find_packages(),
      install_requires=[
          'numpy',
          "mahotas>=1.4.0",
          "matplotlib>=1.4.3",
          "numpy>=1.10.0b1",
          "opencv_python>=2.4.12",
          "pandas>=0.16.2",
          "Pillow>=2.9.0",
          "PIMS>=0.2.2",
          "scikit_image>=0.11.3",
          "scipy>=0.16.0",
          "tifffile>=2015.8.17",
          "trackpy>=0.2.4"
      ],
      scripts=[
            'multiwormtracker/multiwormtracker_app.py',
            ],
      # test_suite='nose.collector',
      # tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False)
