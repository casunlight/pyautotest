# encoding: utf8

"""
Created on 2017.01.16

@author: yalei
"""


from setuptools import setup
from autotest import __version__


setup(name='autotest',
      version=__version__,
      description='Python3 Autotest for nycdatascience',
      url='http://github.com/casunlight/AutoTestPy',
      author='Aiko, Yalei, Shu Yan',
      author_email='aiko.liu@nycdatascience.com',
      packages=['autotest'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
