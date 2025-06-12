from setuptools import setup, find_packages

setup(
    name='pygtfcode',
    version='0.1.0',
    author='Yarone Tokayer',
    description='A Python implementation of the gravothermal fluid code',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yaronetokayer/pygtfcode',  # Update this if/when you publish
    packages=find_packages(),
    install_requires=[
        'numpy>=2.3',
        'scipy>=1.15',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # or your choice
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
