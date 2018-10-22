from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "absl-py==0.2.2",
    "astor==0.6.2",
    "bleach==1.5.0",
    "gast==0.2.0",
    "grpcio==1.12.1",
    "h5py==2.8.0",
    "html5lib==0.9999999",
    "Keras==2.2.0",
    "Keras-Applications==1.0.2",
    "Keras-Preprocessing==1.0.1",
    "Markdown==2.6.11",
    "numpy==1.14.5",
    "protobuf==3.6.0",
    "PyYAML==3.12",
    "six==1.11.0",
    "tensorboard==1.8.0",
    "tensorflow==1.8.0",
    "termcolor==1.1.0",
    "Werkzeug==0.14.1"
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Video Super resolution neural network'
)