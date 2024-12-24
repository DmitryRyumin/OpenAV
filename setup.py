import sys

from setuptools import setup, find_packages

MIN_PYTHON_VERSION = (3, 10)

if sys.version_info[:2] < MIN_PYTHON_VERSION:
    raise RuntimeError("Python version required = {}.{}".format(MIN_PYTHON_VERSION[0], MIN_PYTHON_VERSION[1]))

import openav

REQUIRED_PACKAGES = [
    "ipython == 8.31.0",
    "colorama == 0.4.6",
    "numpy == 1.26.4",
    "pandas == 2.2.3",
    "prettytable == 3.12.0",
    "torch == 2.2.2",
    "torchaudio == 2.2.2",
    "torchvision == 0.17.2",
    "av == 14.0.1",
    "filetype == 1.2.0",
    "vosk == 0.3.44",
    "requests == 2.32.3",
    "pyyaml == 6.0.2",
    "streamlit == 1.41.1",
    "watchdog == 6.0.0",
    "pymediainfo == 6.1.0",
    "pillow == 11.0.0",
    "imgaug == 0.4.0",
    "ffmpeg == 1.4",
    "librosa == 0.10.2.post1",
    "matplotlib == 3.10.0",
    "mediapipe == 0.10.20",
    "opencv_contrib_python == 4.10.0.84",
    "einops == 0.8.0",
    "lion_pytorch == 0.2.3",
    "scikit-learn == 1.6.0",
    "tqdm == 4.67.1",
    "seaborn == 0.13.2",
    "Flask == 3.1.0",
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
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3.12
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
