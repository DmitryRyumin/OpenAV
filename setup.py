import sys

from setuptools import setup, find_packages

MIN_PYTHON_VERSION = (3, 9)

if sys.version_info[:2] < MIN_PYTHON_VERSION:
    raise RuntimeError("Python version required = {}.{}".format(MIN_PYTHON_VERSION[0], MIN_PYTHON_VERSION[1]))

import openav

REQUIRED_PACKAGES = [
    "ipython >= 8.10.0",
    "colorama >= 0.4.6",
    "numpy >= 1.24.2",
    "pandas >= 1.5.3",
    "prettytable >= 3.6.0",
    "torch >= 1.13.1",
    "torchaudio >= 0.13.1",
    "torchvision >= 0.14.1",
    "av >= 10.0.0",
    "filetype >= 1.2.0",
    "vosk >= 0.3.44",
    "requests >= 2.28.2",
    "pyyaml >= 6.0",
    "streamlit >= 1.20.0",
    "watchdog >= 2.3.1",
    "pymediainfo >= 6.0.1",
    "pillow >= 9.5.0",
    "imgaug >= 0.4.0",
    "ffmpeg >= 1.4",
    "librosa >= 0.10.1",
    "matplotlib >= 3.6.3",
    "mediapipe == 0.9.3.0",
    "opencv_contrib_python >= 4.9.0.80",
    "einops >= 0.7.0",
    "lion_pytorch >= 0.1.4",
    "scikit-learn >= 1.4.2",
    "tqdm >= 4.66.2",
    "Seaborn >= 0.13.2",
    "Flask >= 3.0.3",
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
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
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

with open("README.md", "r") as fh:
    long_description = fh.read()

    setup(
        name=openav.__name__,
        packages=find_packages(),
        license=openav.__license__,
        version=openav.__release__,
        author=openav.__author__en__,
        author_email=openav.__email__,
        maintainer=openav.__maintainer__en__,
        maintainer_email=openav.__maintainer_email__,
        url=openav.__uri__,
        description=openav.__summary__,
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=REQUIRED_PACKAGES,
        keywords=[
            "OpenAV",
            "LipReading",
            "SpeechRecognition",
            "SignalProcessing",
            "DataAugmentation",
            "ArtificialNeuralNetworks",
            "DeepMachineLearning",
            "TransferLearning",
            "Statistics",
            "ComputerVision",
            "ArtificialIntelligence",
            "Preprocessing",
        ],
        include_package_data=True,
        classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
        python_requires=">=3.9, <4",
        entry_points={
            "console_scripts": [
                "openav_vad = openav.api.vad:main",
                "openav_vosk_sr = openav.api.vosk_sr:main",
                "openav_preprocess_audio = openav.api.preprocess_audio:main",
                "openav_preprocess_video = openav.api.preprocess_video:main",
                "openav_augmentation = openav.api.augmentation:main",
                "openav_download = openav.api.download:main",
                "openav_train_audiovisual = openav.api.train_audiovisual:main",
                "openav_test_audiovisual = openav.api.test_audiovisual:main",
            ],
        },
        project_urls={
            "Bug Reports": "https://github.com/DmitryRyumin/openav/issues",
            "Documentation": "https://openav.readthedocs.io",
            "Source Code": "https://github.com/DmitryRyumin/openav/tree/main/openav",
            "Download": "https://github.com/DmitryRyumin/openav/tags",
        },
    )
