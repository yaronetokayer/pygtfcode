from setuptools import setup, find_packages

setup(
    name='pygtfcode',
    version='0.1.0',
    author='Yarone Tokayer',
    description='A Python implementation of the gravothermal fluid code',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yaronetokayer/pygtfcode',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.3',
        'scipy>=1.15',
        'numba>=0.61'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
