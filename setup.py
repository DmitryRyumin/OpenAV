import sys

from setuptools import setup, find_packages

MIN_PYTHON_VERSION = (3, 9)

if sys.version_info[:2] < MIN_PYTHON_VERSION:
    raise RuntimeError('Python version required = {}.{}'.format(MIN_PYTHON_VERSION[0], MIN_PYTHON_VERSION[1]))

import openav

REQUIRED_PACKAGES = [
    'ipython >= 8.7.0',
]

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Natural Language :: Russian
Natural Language :: English
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: End Users/Desktop
Intended Audience :: Science/Research
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Image Processing
Topic :: Scientific/Engineering :: Image Recognition
Topic :: Software Development
Topic :: Software Development :: Libraries
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Documentation
Topic :: Documentation :: Sphinx
Topic :: Multimedia :: Sound/Audio
Topic :: Multimedia :: Sound/Audio :: Analysis
Topic :: Multimedia :: Sound/Audio :: Speech
Topic :: Software Development :: Libraries
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Software Development :: Localization
Topic :: Utilities
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Framework :: Jupyter
Framework :: Jupyter :: JupyterLab :: 4
Framework :: Sphinx
"""

with open('README.md', 'r') as fh:
    long_description = fh.read()

    setup(
        name = openav.__name__,
        packages = find_packages(),
        license = openav.__license__,
        version = openav.__release__,
        author = openav.__author__en__,
        author_email = openav.__email__,
        maintainer = openav.__maintainer__en__,
        maintainer_email = openav.__maintainer_email__,
        url = openav.__uri__,
        description = openav.__summary__,
        long_description = long_description,
        long_description_content_type = 'text/markdown',
        install_requires=REQUIRED_PACKAGES,
        keywords = [
            'OpenAV', 'LipReading', 'SpeechRecognition', 'SignalProcessing', 'DataAugmentation',
            'ArtificialNeuralNetworks', 'DeepMachineLearning', 'TransferLearning', 'Statistics', 'ComputerVision',
            'ArtificialIntelligence', 'Preprocessing'
        ],
        include_package_data = True,
        classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
        python_requires = '>=3.9, <4',
        entry_points = {
            'console_scripts': [],
        },
        project_urls = {
            'Bug Reports': 'https://github.com/DmitryRyumin/openav/issues',
            'Documentation': 'https://openav.readthedocs.io',
            'Source Code': 'https://github.com/DmitryRyumin/openav/tree/main/openav',
            'Download': 'https://github.com/DmitryRyumin/openav/tags',
        },
    )
