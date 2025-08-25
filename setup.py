from setuptools import setup, find_packages

setup(
    name='pygtfcode',
    version='2.0.0b1',
    author='Yarone Tokayer',
    description='A Python implementation of a gravothermal fluid code',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yaronetokayer/pygtfcode',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.3',
        'scipy>=1.15',
        'numba>=0.61',
        'matplotlib>=3.10',
        'tqdm>=4.67'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta'
    ],
    python_requires='>=3.12',
)
