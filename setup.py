# coding: utf-8

# The MIT License (MIT)
# 
# Copyright (c) <2024> <Shibzukhov Zaur, szport at gmail dot com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import sys as _sys
_PY36 = _sys.version_info[:2] >= (3, 6)
_PY3 = _sys.version_info[0] >= 3

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

# extra_compile_args = ["-O3",]
extra_compile_args = []

use_cython = 0

if use_cython:
    from Cython.Distutils import Extension, build_ext
    from Cython.Compiler import Options
    Options.fast_fail = True

ext_modules = [
#    Extension(
#        "mltools",
#        ["lib/spn/utils.pyx"],
#    ),
]


description = """mltools: tools for machine learning"""


packages = ['mltools', ]

setup(
    name = 'mltools',
    version = '0.2',
    description = description,
    author = 'Zaur Shibzukhov',
    author_email = 'szport@gmail.com',
    maintainer = 'Zaur Shibzukhov',
    maintainer_email = 'szport@gmail.com',
    license = "MIT License",
    # cmdclass = {'build_ext': build_ext},
    # ext_modules = ext_modules,
    package_dir = {'': 'lib'},
    packages = packages,
    url = 'https://bitbucket.org/intellimath/mltools',
    download_url = 'https://pypi.org/project/mltools/#files',
    long_description="",
    long_description_content_type='text/markdown',
    description_content_type='text/plain',
    platforms='Linux, Mac OS X, Windows',
    keywords=['machine learning', 'data analysis'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
