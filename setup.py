from setuptools import setup, find_packages

setup(
    name='pygtfcode',
    version='1.0.0',
    author='Yarone Tokayer',
    description='A Python implementation of a gravothermal fluid code for SIDM halos',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yaronetokayer/pygtfcode',
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2,<2.3",
        "scipy>=1.15",       # SciPy 1.15 works with NumPy 2.x
        "numba>=0.61.2",     # pairs with NP<=2.2
        'matplotlib>=3.10',
        'tqdm>=4.67'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
